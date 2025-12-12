class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        """
        features: Tensor (N, 512) - Precomputed CLIP embeddings
        targets: Tensor (N,) - Labels
        """
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



def get_cifar10_features(root="./data", batch_size=128, device=cfg.device):
    print("-------------------------------------------------")
    print("1. Loading CLIP model for Feature Extraction...")
    print("-------------------------------------------------")
    # Load model to GPU for fast extraction
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # -------------------------------------------------
    # 2. Load Raw CIFAR with CLIP Preprocess
    # -------------------------------------------------
    # We apply the transform HERE, so the images are ready for the encoder
    train_data = datasets.CIFAR10(root=root, train=True, download=False, transform=preprocess)
    test_data  = datasets.CIFAR10(root=root, train=False, download=False, transform=preprocess)
    
    # Combine them into one temporary loader
    full_raw_dataset = torch.utils.data.ConcatDataset([train_data, test_data])
    loader = DataLoader(full_raw_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Extracting features for {len(full_raw_dataset)} images...")

    # -------------------------------------------------
    # 3. Extraction Loop (Run CLIP Encoder)
    # -------------------------------------------------
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # This is the heavy lifting: Image -> 512-dim Vector
            features = model.encode_image(images)
            
            # Move back to CPU to save GPU memory for training later
            all_features.append(features.cpu())
            all_labels.append(labels)
            
    # Concatenate all batches into single large tensors
    features_tensor = torch.cat(all_features)
    targets_tensor = torch.cat(all_labels).long()

    print(f"Extraction Complete. Feature Shape: {features_tensor.shape}")

    # -------------------------------------------------
    # 4. Cleanup and Return
    # -------------------------------------------------
    # Delete CLIP model to free up GPU memory
    del model
    torch.cuda.empty_cache()

    # Return the lightweight dataset
    return FeatureDataset(features_tensor, targets_tensor)

