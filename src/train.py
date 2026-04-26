#!/usr/bin/env python3
"""
Modular training script for AI vs Real image classifier.
Supports full terminal configuration via argparse.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class SafeImageFolder(Dataset):
    """ImageFolder wrapper that skips corrupted images."""
    def __init__(self, root: str, transform=None):
        self.base_dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        # Pre-filter valid images
        self.valid_samples = []
        for path, label in self.base_dataset.samples:
            try:
                with Image.open(path) as img:
                    img.verify()
                self.valid_samples.append((path, label))
            except Exception:
                print(f"[!] Skipping corrupted image: {path}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx
    
    @property
    def classes(self):
        return self.base_dataset.classes
    
    def __getitem__(self, idx):
        path, label = self.valid_samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class SubsetWithTransform(Dataset):
    """Subset of a dataset that applies a different transform."""
    def __init__(self, base_dataset: SafeImageFolder, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        path, label = self.base_dataset.valid_samples[sample_idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def parse_args() -> argparse.Namespace:
    """Parse all training arguments."""
    parser = argparse.ArgumentParser(
        description="Train a binary classifier to detect AI-generated images vs. Real images."
    )
    # Data paths
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Root directory containing 'real' and 'ai' folders")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    
    # Model architecture
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "custom"],
                        help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate for custom model")
    
    # Data & augmentation
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data for training (rest for validation)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    
    # Class imbalance handling
    parser.add_argument("--real_weight", type=float, default=1.5,
                        help="Loss weight for the 'Real' class (higher = fewer false positives)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for classifying as AI")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # System
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cuda, cpu")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """Determine compute device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    """Build image transforms with robust augmentation for training."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    """Construct the classifier model."""
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> Dict[str, float]:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Class-specific metrics for debugging
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "precision_real": prec_per_class[0],  # class 0 = Real
        "recall_real": rec_per_class[0],
        "precision_ai": prec_per_class[1],   # class 1 = AI
        "recall_ai": rec_per_class[1],
    }


def print_metrics(metrics: Dict, split: str = "Validation"):
    """Pretty-print metrics to terminal."""
    cm = metrics["confusion_matrix"]
    print(f"\n{'='*60}")
    print(f"  {split} Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision (AI):  {metrics['precision']:.4f}")
    print(f"  Recall (AI):     {metrics['recall']:.4f}")
    print(f"  F1-Score (AI):   {metrics['f1_score']:.4f}")
    print(f"{'-'*60}")
    print(f"  Precision (Real): {metrics['precision_real']:.4f}  <- protect this")
    print(f"  Recall (Real):    {metrics['recall_real']:.4f}")
    print(f"{'-'*60}")
    print(f"  Confusion Matrix:")
    print(f"                    Pred:Real  Pred:AI")
    print(f"  Actual:Real        {cm[0,0]:6d}     {cm[0,1]:6d}   (FP={cm[0,1]})")
    print(f"  Actual:AI          {cm[1,0]:6d}     {cm[1,1]:6d}   (FN={cm[1,0]})")
    print(f"{'='*60}\n")


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5) -> Tuple[float, Dict]:
    """Evaluate model on a dataset. Returns (avg_loss, metrics_dict)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            # Apply threshold to class 1 (AI) probability
            preds = (probs[:, 1] >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), class_names=["Real", "AI"])
    return running_loss / len(loader), metrics


def save_checkpoint(model: nn.Module, epoch: int, metrics: Dict, path: Path):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"[OK] Saved checkpoint: {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data directory: {data_dir.absolute()}")
    print(f"[INFO] Output directory: {output_dir.absolute()}")
    
    # Load dataset
    train_transform = get_transforms(args.img_size, is_train=True)
    val_transform = get_transforms(args.img_size, is_train=False)
    
    full_dataset = SafeImageFolder(root=str(data_dir), transform=train_transform)
    
    # Map folder names to class indices
    class_to_idx = full_dataset.class_to_idx  # e.g., {'ai': 0, 'real': 1} or vice versa
    print(f"[INFO] Classes found: {class_to_idx}")
    
    # Split dataset
    n_train = int(len(full_dataset) * args.train_split)
    n_val = len(full_dataset) - n_train
    indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_ds = SubsetWithTransform(full_dataset, train_indices, transform=train_transform)
    val_ds = SubsetWithTransform(full_dataset, val_indices, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    print(f"[INFO] Train samples: {n_train}, Val samples: {n_val}")
    
    # Build model
    model = build_model(args.model, num_classes=2, pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device)
    
    # Loss with class weighting: higher weight for Real class reduces false positives
    # class_to_idx: real->idx, ai->idx
    real_idx = class_to_idx.get("real", 0)
    ai_idx = class_to_idx.get("ai", 1)
    weights = torch.tensor([args.real_weight if real_idx == 0 else 1.0,
                            args.real_weight if real_idx == 1 else 1.0]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                     patience=3)
    
    print(f"[INFO] Model: {args.model}, Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"[INFO] Real class weight: {args.real_weight}, Threshold: {args.threshold}")
    print("="*60)
    
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)
        
        scheduler.step(val_metrics["f1_score"])
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{args.epochs} | {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f}")
        
        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"{args.model}_epoch{epoch}.pt"
            save_checkpoint(model, epoch, val_metrics, ckpt_path)
        
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_path = output_dir / f"{args.model}_best.pt"
            save_checkpoint(model, epoch, val_metrics, best_path)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION (Best Model)")
    print("="*60)
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, final_metrics = evaluate(model, val_loader, device, threshold=args.threshold)
    print_metrics(final_metrics, split="Validation")
    
    print(f"[OK] Training complete. Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
