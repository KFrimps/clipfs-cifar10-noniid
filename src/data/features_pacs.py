import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import clip
from PIL import Image
import numpy as np

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

class HFImageDataset(Dataset):
    """
    Wrapper to make Hugging Face dataset compatible with Torch DataLoader 
    and CLIP preprocessing.
    """
    def __init__(self, hf_data, transform=None):
        self.hf_data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = item['image']  # Hugging Face datasets use 'image' key
        label = item['label']  # and 'label' key
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_pacs_features_by_domain(batch_size=128):
    """
    Downloads PACS from Hugging Face, filters by domain, runs CLIP *once*, 
    and returns a lightweight dataset with domain indices.
    """
    print("Loading CLIP for Feature Extraction...")
    model, preprocess = clip.load("ViT-B/32", device=cfg.device)
    model.eval()

    print("Downloading PACS from Hugging Face...")
    # 1. Load the Entire Dataset
    # flwrlabs/pacs typically has only a 'train' split containing all domains
    full_hf_dataset = load_dataset("flwrlabs/pacs", split="train")
    
    # 2. Define Domains
    # These names must match the values in the 'domain' column of the dataset
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    domain_indices = {} 
    
    all_features = []
    all_labels = []
    
    current_idx = 0
    
    # 3. Iterate through each domain by FILTERING the main dataset
    for domain_name in domains:
        print(f"--- Processing Domain: {domain_name} ---")
        
        # Filter the dataset for the specific domain
        # This creates a subset view without duplicating data in memory
        domain_data = full_hf_dataset.filter(lambda x: x['domain'] == domain_name)
        
        if len(domain_data) == 0:
            print(f"Warning: No images found for domain '{domain_name}'. Check spelling.")
            continue

        # Wrap it for PyTorch
        dataset = HFImageDataset(domain_data, transform=preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Extracting features for {domain_name} ({len(dataset)} images)...")
        
        domain_start = current_idx
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                
                # Extract features
                features = model.encode_image(images)
                
                all_features.append(features.cpu())
                all_labels.append(labels)
                
                current_idx += len(labels)
        
        domain_end = current_idx
        
        # Save the index range for this domain
        domain_indices[domain_name] = list(range(domain_start, domain_end))

    # 4. Concatenate all to create the master tensors
    if not all_features:
        raise RuntimeError("No features extracted. The dataset filter likely returned empty results.")

    features_tensor = torch.cat(all_features)
    targets_tensor = torch.cat(all_labels).long()
    
    # Cleanup to free GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Create the lightweight dataset
    full_dataset = FeatureDataset(features_tensor, targets_tensor)
    
    return full_dataset, domain_indices
