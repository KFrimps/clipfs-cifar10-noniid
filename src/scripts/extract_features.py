# scripts/extract_features.py

"""
Extract CLIP features for CIFAR-10 and save them to disk.
Run this script ONCE before federated training.

Usage:
    python scripts/extract_features.py
"""

import torch
from pathlib import Path
from torchvision import datasets, transforms
from src.data.features import extract_clip_features
from src.utils.paths import FEATURES_DIR
from src.config import CFG

def main():
    cfg = CFG()
    device = cfg.device

    # --------------------------
    # 1. Load CIFAR-10 images
    # --------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # CLIP expects 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading CIFAR-10 dataset...")
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # --------------------------
    # 2. Extract CLIP features
    # --------------------------
    print("Extracting CLIP features...")
    features, labels = extract_clip_features(
        dataset,
        batch_size=cfg.batch_size,
        device=device
    )

    # --------------------------
    # 3. Save to disk
    # --------------------------
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = FEATURES_DIR / "cifar10_clip_features.pt"
    torch.save(
        {
            "features": features.cpu(),
            "labels": labels.cpu(),
        },
        out_path
    )

    print(f"Saved CLIP features to:\n{out_path}")

if __name__ == "__main__":
    main()
