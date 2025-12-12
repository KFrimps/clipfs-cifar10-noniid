# clipfs-noniid

Federated Learning with CLIP Features Under Non-IID and Domain-Shifted Scenarios

This repository contains a unified and modular implementation of federated learning (FL) using frozen CLIP image embeddings and lightweight linear classifier heads.
It supports multiple experimental setups, including:

CIFAR-10 General Non-IID
(Quantity skew + Label skew using Dirichlet distributions)

CIFAR-10 Extreme Non-IID
(Strict one-class-per-client splits / heavy distribution shift)

PACS Domain-Shift Federated Learning
(Clients correspond to domains: Photo → Cartoon → Sketch → Art painting)

The goal is to study how CLIP’s pretrained representations behave under heterogeneous federated environments, how non-IID distributions impact global learning, and how simple classifier heads can converge efficiently using Flower.

🔥 Key Features
1. CLIP as a Frozen Feature Extractor

All datasets are converted into 512-dim CLIP (ViT-B/32) embeddings once.
This removes compute bottlenecks and isolates the effect of data heterogeneity on model aggregation, as done in the FedCLIP and PromptFL literature.

2. Modular Data Splits

The repo provides reusable splitting functions:

Quantity + Label Skew (Dirichlet) for CIFAR-10

Extreme One-Class Non-IID

Few-Shot sampling

Domain-based splitting for PACS

3. Lightweight FL Training

Each client trains only a small linear classifier head, dramatically reducing:

compute cost

communication cost

memory footprint

4. Full Logging for Analysis

A custom FedAvg strategy logs:

global metrics

per-client metrics

hyperparameter tuning outcomes

CSV logs compatible with pandas/Excel/Matplotlib

5. Clean, Reusable Project Structure

All shared FL logic, splitting logic, feature extractors, and training utilities live in src/, while each experiment is a small script inside src/experiments/.

📂 Project Structure
fedclip-noniid-project/
│
├─ src/
│  ├─ config/                   # Experiment-specific settings
│  ├─ data/                     # CLIP feature loaders + split functions
│  ├─ models/                   # Linear classifier heads
│  ├─ fl_core/                  # Clients, strategy, tuning
│  ├─ experiments/              # 3 runnable experiments
│  └─ utils/                    # Seed control, plotting, paths
│
├─ scripts/
│  ├─ extract_features.py       # One-time CLIP feature extraction
│  └─ tune_hparams.py           # Optional hyperparameter tuning
│
├─ runs/                        # CSV logs + plots (git-ignored)
├─ requirements.txt
└─ README.md

🚀 Getting Started
1. Install dependencies
pip install -r requirements.txt


Ensure Torch, torchvision, and Flower are installed.

2. Extract CLIP features (run once per dataset)
CIFAR-10
python scripts/extract_features.py --dataset cifar10

PACS
python scripts/extract_features.py --dataset pacs


This saves precomputed embeddings in:

data/features/
    cifar10_clip_features.pt
    pacs_clip_features.pt

3. Run Experiments
CIFAR-10: General Non-IID (Quantity + Label Skew)
python -m src.experiments.run_cifar10_general

CIFAR-10: Extreme One-Class Non-IID
python -m src.experiments.run_cifar10_extreme

PACS: Domain-Shift Federated Learning
python -m src.experiments.run_pacs_domain_shift


Results will automatically be saved in:

runs/<experiment-name>/
    global_metrics.csv
    client_metrics.csv
    eval_client_metrics.csv
    plots/

⚙️ Hyperparameter Tuning (Optional)

A tuning pipeline is provided for quickly sweeping learning rates, local epochs, etc.

Run:

python scripts/tune_hparams.py


This evaluates models on a subset of clients and writes best hyperparameters to:

runs/hparam_results.json


Experiment scripts will automatically consume these values if present.

📊 Visualizations

The repo includes helper utilities to generate:

Class distribution histograms per client

Non-IID heatmaps

Accuracy vs communication rounds

Per-client evaluation curves

Plots are automatically saved under:

runs/<experiment>/plots/

🧠 Research Motivation

This repository supports experiments investigating:

How non-IID distributions degrade global model accuracy

How client-level diversity affects convergence

Whether frozen CLIP features provide robustness under distribution shift

How few-shot client updates impact personalization

How extreme heterogeneity (one-class clients) influences global aggregation

It aligns with concepts from FedCLIP, PromptFL, CReFF, and domain generalization literature.

📚 Referenced Papers

While this project does not re-implement the full FedCLIP architecture, it follows the scientific motivation of:

FedCLIP (Lu et al., 2023)

CLIP-Guided FL (Shi et al., 2024)

PromptFL (Wang et al., 2022)

CReFF (Shang et al., 2022)

Flower: A Friendly Federated Learning Framework

🤝 Contributing

PRs for:

new datasets

additional non-IID modes

personalized FL extensions

CLIP text-encoder integration

…are welcome.

📄 License

MIT License.
You are free to use, extend, and publish results derived from this project.

🏁 Final Note

This repository is built for clarity, reproducibility, and research extensibility.
If you are studying:

federated learning under real-world data heterogeneity

foundation models in distributed settings

domain generalization and cross-client robustness

…this codebase gives you a strong, modular foundation.
