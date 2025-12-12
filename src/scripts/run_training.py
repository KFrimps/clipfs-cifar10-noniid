# scripts/run_training.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import flwr as fl
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig
from flwr.server.app import ServerAppComponents
from flwr.simulation import run_simulation

from src.config import CFG
from src.utils.seed import set_seed
from src.utils.paths import FEATURES_DIR
from src.data.features import FeatureDataset
from src.data.partition import quantity_and_label_skew_split, split_client_train_test
from src.models.clip_head import make_model
from src.fl.client import CIFARClient
from src.fl.server import LogGlobalEvalFedAvg, make_metric_logger


def main():
    # -------------------------
    # 0. Config + seed
    # -------------------------
    cfg = CFG()
    set_seed(cfg.seed)

    # -------------------------
    # 1. Load CLIP features
    #    (from extract_features.py)
    # -------------------------
    feat_path = FEATURES_DIR / "cifar10_clip_features.pt"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run scripts/extract_features.py first."
        )

    data = torch.load(feat_path)
    features = data["features"]  # [N, 512]
    labels = data["labels"]      # [N]

    full_dataset = FeatureDataset(features, labels)
    print(f"Loaded feature dataset: {len(full_dataset)} samples")

    # -------------------------
    # 2. Split indices into clients
    #    (quantity + label skew)
    # -------------------------
    print("Splitting data into clients (quantity + label skew)...")

    client_parts = quantity_and_label_skew_split(
        trainset=full_dataset,
        k=cfg.clients,
        alpha_qty=cfg.alpha_qty,      # e.g. 1.0
        alpha_label=cfg.alpha_label,  # e.g. 0.1
        min_per_client=cfg.min_per_client,
        seed=cfg.seed,
        max_tries=50,
    )

    # Now split each client into train/test indices
    client_train_parts = []
    client_test_parts  = []

    for cid, idxs in enumerate(client_parts):
        train_idx, test_idx = split_client_train_test_strict(
            idxs,
            full_dataset=full_dataset,
            test_frac=0.2,
            seed=cid,
            num_classes=10
        )

        # FEW-SHOT PART 
        fewshot_train_idx = make_fewshot(
            train_idx,
            full_dataset=full_dataset,
            shots_per_class=7
        )

        client_train_parts.append(fewshot_train_idx)          
        client_test_parts.append(test_idx)


    tune_global_hyperparams(
        full_dataset=full_dataset,
        client_train_parts=client_train_parts,
        cfg=cfg,
        make_model_fn=make_model,   # same function you use in client_fn
    )


    # -------------------------
    # 3. Build DataLoaders per client
    # -------------------------
    client_train_loaders = []
    client_test_loaders = []

    for cid, train_idx in enumerate(client_train_parts):
        tr_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        te_loader = DataLoader(
            Subset(full_dataset, client_test_parts[cid]),
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        client_train_loaders.append(tr_loader)
        client_test_loaders.append(te_loader)

    # -------------------------
    # 4. client_fn: what each client does
    # -------------------------
    def client_fn(context: fl.common.Context):
        part_id = int(context.node_config["partition-id"])

        model = make_model()  # small linear head

        return CIFARClient(
            model=model,
            train_loader=client_train_loaders[part_id],
            test_loader=client_test_loaders[part_id],
            cfg=cfg,
            client_idx=part_id,
        ).to_client()

    # -------------------------
    # 5. Strategy & Server
    # -------------------------
    init_model = make_model()
    init_params = [v.cpu().numpy() for v in init_model.state_dict().values()]
    initial_parameters = fl.common.ndarrays_to_parameters(init_params)

    strategy = LogGlobalEvalFedAvg(
        # which clients train each round
        fraction_fit=cfg.client_fraction,
        min_fit_clients=cfg.clients,
        min_available_clients=cfg.clients,

        # ask all clients to evaluate every round
        fraction_evaluate=1.0,
        min_evaluate_clients=cfg.clients,

        # push local epochs to clients
        on_fit_config_fn=lambda rnd: {"local_epochs": int(cfg.local_epochs)},

        # aggregate client evals + log CSV
        evaluate_metrics_aggregation_fn=make_metric_logger(),

        # initial global weights
        initial_parameters=initial_parameters,
    )

    client_app = ClientApp(client_fn=client_fn)

    def server_fn(context: fl.common.Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=cfg.rounds),
        )

    server_app = ServerApp(server_fn=server_fn)

    # -------------------------
    # 6. Run simulation
    # -------------------------
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.clients,
        backend_name="ray",
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": 0.2,  # small fraction, only for linear head
            }
        },
    )


if __name__ == "__main__":
    main()
