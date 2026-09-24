import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PACSDataset(Dataset):
    def __init__(self, data_root, domain, split="all", split_file="shared/splits/pacs_sketch_seed6304.json", transform=None):
        """
        domain: 'photo', 'art_painting', 'cartoon', or 'sketch'
        split: 'train', 'val', or 'all'
        """
        self.data_root = Path(data_root) / domain
        self.domain = domain
        self.transform = transform
        self.classes = sorted(["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = [] # list of (path, label)
        
        if split == "all":
            # Load all images in the domain directory
            for cls in self.classes:
                cls_dir = self.data_root / cls
                if not cls_dir.exists(): continue
                for img_name in os.listdir(cls_dir):
                    self.samples.append((cls_dir / img_name, self.class_to_idx[cls]))
        else:
            # Load from split file
            with open(split_file, "r") as f:
                splits = json.load(f)
            
            for rel_path in splits[domain][split]:
                cls = rel_path.split("/")[0]
                self.samples.append((self.data_root / rel_path, self.class_to_idx[cls]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms():
    """
    Returns (train_transform, eval_transform)
    Resize to 256x256, 224x224 crop, horizontal flip for train, center crop for val.
    Normalize using ImageNet weights.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, eval_transform

class CombinedSourceLoader:
    """Cycles through multiple data loaders so that a batch contains equal examples from each source."""
    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max(len(loader) for loader in loaders)
        
    def __iter__(self):
        self.iters = [iter(loader) for loader in self.loaders]
        self.step = 0
        return self
        
    def __next__(self):
        if self.step >= self.max_batches:
            raise StopIteration
            
        self.step += 1
        batch_x = []
        batch_y = []
        for i in range(len(self.iters)):
            try:
                x, y = next(self.iters[i])
            except StopIteration:
                self.iters[i] = iter(self.loaders[i])
                x, y = next(self.iters[i])
            batch_x.append(x)
            batch_y.append(y)
            
        return torch.cat(batch_x, dim=0), torch.cat(batch_y, dim=0)

def get_task2_loaders(data_root="data/PACS", batch_size_per_source=8, target_batch_size=24):
    train_tf, eval_tf = get_transforms()
    
    # Source loaders
    source_domains = ["photo", "art_painting", "cartoon"]
    train_loaders = []
    val_loaders = {}
    
    for domain in source_domains:
        train_ds = PACSDataset(data_root, domain, split="train", transform=train_tf)
        val_ds = PACSDataset(data_root, domain, split="val", transform=eval_tf)
        
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size_per_source, shuffle=True, drop_last=True, num_workers=2, pin_memory=True))
        val_loaders[domain] = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
        
    combined_source_loader = CombinedSourceLoader(train_loaders)
    
    # Target loader (Sketch)
    target_ds = PACSDataset(data_root, "sketch", split="all", transform=train_tf) # Training transform for target during UDA
    target_loader = DataLoader(target_ds, batch_size=target_batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
    # Final eval target loader
    target_eval_ds = PACSDataset(data_root, "sketch", split="all", transform=eval_tf)
    target_eval_loader = DataLoader(target_eval_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    return combined_source_loader, val_loaders, target_loader, target_eval_loader
