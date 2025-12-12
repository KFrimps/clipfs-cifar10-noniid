# clipfs-noniid

# Mitigating Non-IID Effects in Federated Learning through CLIP-based Linear Probing.

This repository provides a unified and modular framework for experimenting with **Federated Learning (FL)** using **frozen CLIP visual embeddings** and **lightweight linear classifier heads**.  
It supports multiple heterogeneity scenarios, including:

- **CIFAR-10 General Non-IID**  
  (Quantity skew + Label skew via Dirichlet distributions)
- **CIFAR-10 Extreme Non-IID**  
  (Strict one-class-per-client setups)
- **PACS Domain Shift Federated Learning**  
  (Clients = domains: Photo, Cartoon, Sketch, Art Painting)

The goal is to analyze how **pretrained CLIP representations** behave under severe data heterogeneity, how distribution shift affects convergence, and how simple models can be trained efficiently on top of strong frozen feature encoders.

---

## Key Features

- **Frozen CLIP (ViT-B/32) embeddings** for fast, compute-efficient FL.
- **Flexible non-IID splitting**:
  - quantity skew
  - label skew
  - extreme one-class splits
  - domain-based splits (PACS)
- **Lightweight classifier head** trained locally on each client.
- **Custom FedAvg strategy** with:
  - global metric logging  
  - per-client evaluation  
  - CSV export for reproducible analysis  
- **Hyperparameter tuning module** included.
- **Modular repository design** for easy extension.

---


