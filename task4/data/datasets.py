import json
from pathlib import Path
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

def get_transforms(method="vanilla"):
    # Base transforms
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    if method == "gcsc":
        # GCSC uses RandAugment
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Vanilla / PROSER
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Unaugmented for Mahalanobis
    unaug_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_tf, eval_tf, unaug_tf

def get_cifar_loaders(data_root="data", method="vanilla", batch_size=128, num_workers=2):
    train_tf, eval_tf, unaug_tf = get_transforms(method)
    
    # Load splits
    split_dir = Path("task4/data")
    with open(split_dir / "cifar10_splits.json", "r") as f:
        c10_splits = json.load(f)
    with open(split_dir / "cifar100_unknowns.json", "r") as f:
        c100_splits = json.load(f)
        
    # CIFAR-10 Datasets
    c10_full_train = CIFAR10(root=data_root, train=True, download=True)
    
    c10_train = Subset(CIFAR10(root=data_root, train=True, transform=train_tf), c10_splits["train"])
    c10_val = Subset(CIFAR10(root=data_root, train=True, transform=eval_tf), c10_splits["val"])
    c10_train_unaug = Subset(CIFAR10(root=data_root, train=True, transform=unaug_tf), c10_splits["train"])
    
    c10_test = CIFAR10(root=data_root, train=False, transform=eval_tf, download=True)
    
    # CIFAR-100 Unknowns Datasets
    c100_test = CIFAR100(root=data_root, train=False, transform=eval_tf, download=True)
    c100_near = Subset(c100_test, c100_splits["near"])
    c100_far = Subset(c100_test, c100_splits["far"])
    
    # Loaders
    loaders = {
        "train": DataLoader(c10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(c10_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "train_unaug": DataLoader(c10_train_unaug, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(c10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "near": DataLoader(c100_near, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "far": DataLoader(c100_far, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    
    return loaders
