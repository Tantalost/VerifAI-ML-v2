#!/usr/bin/env python3
"""
Model comparison script for AI vs Real image classifier.
Loads two checkpoints and prints side-by-side metrics on a held-out test set.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from train import build_model, set_seed, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two trained models on a test set."
    )
    parser.add_argument("model_a", type=str, help="Path to first model checkpoint (.pt)")
    parser.add_argument("model_b", type=str, help="Path to second model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Root directory containing 'real' and 'ai' folders")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "custom"],
                        help="Model architecture used during training")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for classifying as AI")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def load_model(checkpoint_path: Path, model_name: str, device: torch.device) -> nn.Module:
    """Load a model from a checkpoint file."""
    model = build_model(model_name, num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                   threshold: float) -> dict:
    """Run inference and compute all metrics."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "precision_real": precision_score(y_true, y_pred, average=None, zero_division=0)[0],
        "recall_real": recall_score(y_true, y_pred, average=None, zero_division=0)[0],
        "confusion_matrix": cm,
        "false_positives": int(cm[0, 1]),   # Real -> AI
        "false_negatives": int(cm[1, 0]),   # AI -> Real
    }


def print_header(title: str, width: int = 70):
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def print_side_by_side(name_a: str, metrics_a: dict, name_b: str, metrics_b: dict):
    """Print side-by-side model comparison in terminal."""
    w = 70
    print_header("MODEL COMPARISON", w)
    
    # Header row
    print(f"  {'Metric':<22} {name_a:<22} {name_b:<22}")
    print("-" * w)
    
    # Metrics rows
    metrics = [
        ("Accuracy", "accuracy", ".4f"),
        ("Precision (AI)", "precision", ".4f"),
        ("Recall (AI)", "recall", ".4f"),
        ("F1-Score (AI)", "f1_score", ".4f"),
        ("Precision (Real)", "precision_real", ".4f"),
        ("Recall (Real)", "recall_real", ".4f"),
        ("False Positives", "false_positives", "d"),
        ("False Negatives", "false_negatives", "d"),
    ]
    
    for label, key, fmt in metrics:
        val_a = metrics_a[key]
        val_b = metrics_b[key]
        winner = ""
        if key not in ("false_positives", "false_negatives"):
            if val_a > val_b:
                winner = "  <<< A"
            elif val_b > val_a:
                winner = "  <<< B"
        else:
            if val_a < val_b:
                winner = "  <<< A (better)"
            elif val_b < val_a:
                winner = "  <<< B (better)"
        print(f"  {label:<22} {val_a:{fmt}:>22} {val_b:{fmt}:>22}{winner}")
    
    print("-" * w)
    
    # Confusion matrices
    cm_a = metrics_a["confusion_matrix"]
    cm_b = metrics_b["confusion_matrix"]
    
    print(f"\n  {'Confusion Matrix A':<35} {'Confusion Matrix B':<35}")
    print(f"  {'Pred:Real  Pred:AI':<35} {'Pred:Real  Pred:AI':<35}")
    print(f"  Real  {cm_a[0,0]:6d}    {cm_a[0,1]:6d}          Real  {cm_b[0,0]:6d}    {cm_b[0,1]:6d}")
    print(f"  AI    {cm_a[1,0]:6d}    {cm_a[1,1]:6d}          AI    {cm_b[1,0]:6d}    {cm_b[1,1]:6d}")
    print("=" * w + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")
    
    # Validate paths
    path_a = Path(args.model_a)
    path_b = Path(args.model_b)
    if not path_a.exists():
        print(f"[ERROR] Model A not found: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"[ERROR] Model B not found: {path_b}")
        sys.exit(1)
    
    # Load models
    print(f"[INFO] Loading Model A: {path_a.name}")
    model_a = load_model(path_a, args.model, device)
    print(f"[INFO] Loading Model B: {path_b.name}")
    model_b = load_model(path_b, args.model, device)
    
    # Build test dataset
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(root=args.data_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"[INFO] Test samples: {len(test_dataset)}")
    print(f"[INFO] Classes: {test_dataset.class_to_idx}")
    
    # Evaluate both models
    print(f"\n[INFO] Evaluating Model A ({path_a.name}) ...")
    metrics_a = evaluate_model(model_a, test_loader, device, args.threshold)
    
    print(f"[INFO] Evaluating Model B ({path_b.name}) ...")
    metrics_b = evaluate_model(model_b, test_loader, device, args.threshold)
    
    # Print comparison
    print_side_by_side(path_a.name, metrics_a, path_b.name, metrics_b)
    
    print("[OK] Comparison complete.")


if __name__ == "__main__":
    main()
