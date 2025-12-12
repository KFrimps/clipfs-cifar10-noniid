from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn

def _validate_loader_global(model, loader, device):
    """Small helper: compute avg loss on a DataLoader."""
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if x.dtype != torch.float32:
                x = x.float()
            if y.dtype != torch.long:
                y = y.long()

            logits = model(x)
            loss_sum += crit(logits, y).item() * x.size(0)
            total += x.size(0)

    if total == 0:
        return 0.0
    return loss_sum / total

def tune_global_hyperparams(full_dataset, client_train_parts, cfg, make_model_fn):
    """
    One-shot global CV + grid search on the *combined client train indices*.

    - full_dataset: your FeatureDataset (CLIP embeddings + labels)
    - client_train_parts: list of np.arrays (few-shot train indices per client)
    - cfg: config object (has lr, momentum, weight_decay, batch_size, device)
    - make_model_fn: callable that returns a fresh classifier model (same as in client_fn)

    This function:
        * concatenates all client train indices (no test leakage),
        * runs K-fold CV for each LR in a small grid,
        * picks best LR and average best epoch,
        * updates cfg.lr and cfg.local_epochs in-place.
    """
    print("\n[Hyperparam Tuning] Starting global CV + grid search...")

    # 1) Build the global train index pool (only TRAIN parts, no test)
    import numpy as np
    all_train_indices = np.concatenate(client_train_parts)
    all_train_indices = np.array(all_train_indices, dtype=int)

    # 2) Define search space (you can tweak these)
    learning_rates_to_try = [0.001, 0.005, 0.01]
    max_epochs = 5          # ceiling for early stopping in each fold
    n_splits = 5            # K-fold

    # 3) Setup KFold over the global train indices
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_lr_overall = cfg.lr          # fallback
    best_avg_loss = float("inf")
    optimal_epochs_for_best_lr = 1    # fallback

    device = cfg.device
    crit = nn.CrossEntropyLoss()

    for lr in learning_rates_to_try:
        fold_val_losses = []
        fold_best_epochs = []

        print(f"[Hyperparam Tuning] Trying LR = {lr}")

        # K-fold loop
        for fold, (train_idx_local, val_idx_local) in enumerate(kf.split(all_train_indices)):
            # Map from fold indices -> actual dataset indices
            fold_train_global = all_train_indices[train_idx_local]
            fold_val_global = all_train_indices[val_idx_local]

            # Create DataLoaders
            train_sub = DataLoader(
                Subset(full_dataset, fold_train_global),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            val_sub = DataLoader(
                Subset(full_dataset, fold_val_global),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            # Fresh model for each fold
            model = make_model_fn().to(device)
            opt = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )

            best_fold_loss = float("inf")
            best_fold_epoch = 0

            # Simple early-stopping style search up to max_epochs
            for epoch in range(max_epochs):
                model.train()
                for x, y in train_sub:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    if x.dtype != torch.float32:
                        x = x.float()
                    if y.dtype != torch.long:
                        y = y.long()

                    opt.zero_grad()
                    logits = model(x)
                    loss = crit(logits, y)
                    loss.backward()
                    opt.step()

                # Validate on this fold
                val_loss = _validate_loader_global(model, val_sub, device)

                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss
                    best_fold_epoch = epoch + 1  # 1-based for readability

            fold_val_losses.append(best_fold_loss)
            fold_best_epochs.append(best_fold_epoch)
            print(f"  Fold {fold}: best_val_loss={best_fold_loss:.4f} at epoch={best_fold_epoch}")

        # Evaluate this LR over all folds
        avg_loss_for_lr = sum(fold_val_losses) / len(fold_val_losses)
        avg_epoch_for_lr = int(sum(fold_best_epochs) / len(fold_best_epochs))

        print(f"[Hyperparam Tuning] LR={lr}: avg_val_loss={avg_loss_for_lr:.4f}, "
              f"avg_best_epoch={avg_epoch_for_lr}")

        # Check if this LR wins
        if avg_loss_for_lr < best_avg_loss:
            best_avg_loss = avg_loss_for_lr
            best_lr_overall = lr
            optimal_epochs_for_best_lr = max(1, avg_epoch_for_lr)

    print(f"\n[Hyperparam Tuning] Winner -> LR={best_lr_overall}, "
          f"epochs={optimal_epochs_for_best_lr}")

    # 4) Update cfg in-place so all clients use these hyperparams
    cfg.lr = best_lr_overall
    cfg.local_epochs = optimal_epochs_for_best_lr
    print(f"[Hyperparam Tuning] cfg.lr set to {cfg.lr}, cfg.local_epochs set to {cfg.local_epochs}")
