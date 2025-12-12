import numpy as np
from collections import Counter

def quantity_and_label_skew_split(
    trainset,
    k,
    alpha_qty,
    alpha_label,
    min_per_client,
    seed,
    max_tries,
):
    """
    Combined quantity + label skew using Dirichlet.

    - alpha_qty   controls how unbalanced total client sizes are
    - alpha_label controls how skewed class proportions are per client
    - min_per_client is a soft constraint; we resample until everyone has at least this many points

    Returns:
        parts: list of length k, each an np.array of indices into trainset
    """
    rng = np.random.default_rng(cfg.seed)
    targets = np.array(trainset.targets)
    N = len(targets)

    classes = np.unique(targets)
    C = len(classes)

    # Indices per class (e.g., CIFAR-10 -> 10 classes, each ~6000 samples)
    idx_by_class = {c: np.where(targets == c)[0].tolist() for c in classes}

    # Try a few times until everyone has >= min_per_client samples
    for _ in range(max_tries):
        # fresh shuffled pools each attempt
        pools = {c: list(idx_by_class[c]) for c in classes}
        for c in classes:
            rng.shuffle(pools[c])

        parts = [[] for _ in range(k)]

        # ▶ 1) Sample "size tendencies" for clients (quantity skew)
        #     Clients with bigger size_props will tend to get more samples from every class
        size_props = rng.dirichlet([alpha_qty] * k)  # shape (k,), sums to 1

        # ▶ 2) For each class, allocate its samples using a Dirichlet
        #      whose parameters are biased by size_props (so big clients get more of every class)
        for c in classes:
            pool = pools[c]
            Nc = len(pool)

            # Dirichlet concentration for this class, biased by size_props
            base_conc = alpha_label * k
            params = base_conc * size_props           # shape (k,)

            p = rng.dirichlet(params)                 # class-c proportions per client
            raw = p * Nc
            base = np.floor(raw).astype(int)
            deficit = Nc - base.sum()

            if deficit > 0:
                frac = raw - base
                bump = np.argsort(-frac)[:deficit]
                base[bump] += 1

            # Now sum_k base[k] == Nc
            start = 0
            for i in range(k):
                take = int(base[i])
                if take > 0:
                    parts[i].extend(pool[start:start + take])
                    start += take

        # Shuffle inside each client and check sizes
        parts = [np.array(rng.permutation(p), dtype=int) for p in parts]
        sizes = [len(p) for p in parts]

        if min(sizes) >= min_per_client:
            # Success: everyone has enough data
            return parts

    raise RuntimeError("Failed to sample a split satisfying min_per_client within max_tries")

def strict_one_class_split(dataset, n_clients):
    """
    Assigns exactly one distinct class to each client.
    Note: If n_clients < n_classes, the remaining classes are DROPPED.
    """
    # Handle targets whether they are Tensor or Numpy
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    unique_classes = np.unique(targets) # [0, 1, 2, 3, 4, 5, 6]
    
    if n_clients > len(unique_classes):
        raise ValueError(f"Cannot assign 1 unique class per client. "
                         f"Clients ({n_clients}) > Classes ({len(unique_classes)})")

    parts = []
    print(f"\n--- STRICT SPLIT: 1 CLASS PER CLIENT ---")
    
    for i in range(n_clients):
        target_cls = unique_classes[i]
        
        # Get all indices where label == target_cls
        cls_idx = np.where(targets == target_cls)[0]
        
        # Shuffle for randomness in train/test later
        np.random.shuffle(cls_idx)
        
        parts.append(cls_idx)
        
        print(f"Client {i} -> Assigned Class {target_cls} only. (n={len(cls_idx)})")
        
    print(f"WARNING: Classes {unique_classes[n_clients:]} were dropped and will be unseen.\n")
    
    return parts
    

def split_client_train_test_strict(idxs, full_dataset, test_frac, seed, num_classes):
    """
    Stratified per client:
    - For this client's indices only (idxs),
    - split per class into train/test with the SAME counts per class 
      across all clients (because each client started from iid_split_strict).
    """
    rng = np.random.default_rng(seed)
    labels = np.array(full_dataset.targets)

    train_idx = []
    test_idx  = []

    for cls in range(num_classes):
        # all this client's indices that belong to class cls
        cls_idx = [i for i in idxs if labels[i] == cls]
        cls_idx = np.array(cls_idx, dtype=int)

        # shuffle only within this class
        rng.shuffle(cls_idx)

        n_c = len(cls_idx)
        n_test_c = int(n_c * test_frac)   # same for all clients since n_c is the same
        n_train_c = n_c - n_test_c

        test_idx.extend(cls_idx[:n_test_c])
        train_idx.extend(cls_idx[n_test_c:])

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def make_fewshot(train_idx, full_dataset, shots_per_class):
    """
    Returns only up to 'shots_per_class' samples per class 
    from this client's training indices.
    """
    labels = np.array(full_dataset.targets)

    selected = []
    for cls in range(10):        # CIFAR-10 has 10 classes
        cls_idx = [i for i in train_idx if labels[i] == cls]
        np.random.shuffle(cls_idx)
        selected.extend(cls_idx[:shots_per_class])  # take first K

    return np.array(selected)
