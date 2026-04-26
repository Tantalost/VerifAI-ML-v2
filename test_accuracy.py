#!/usr/bin/env python3
"""
Quick test script to evaluate best.pt model accuracy on the dataset.
Run this from PowerShell: python test_accuracy.py
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class SafeImageFolder(Dataset):
    """ImageFolder wrapper that skips corrupted images."""
    def __init__(self, root, transform=None):
        self.base_dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.valid_samples = []
        for path, label in self.base_dataset.samples:
            try:
                with Image.open(path) as img:
                    img.verify()
                self.valid_samples.append((path, label))
            except Exception:
                print(f"Skipping corrupted: {path}")

    def __len__(self):
        return len(self.valid_samples)

    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx

    def __getitem__(self, idx):
        path, label = self.valid_samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(model_name="resnet18", num_classes=2, dropout=0.5):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "precision_real": precision_score(y_true, y_pred, average=None, zero_division=0)[0],
        "recall_real": recall_score(y_true, y_pred, average=None, zero_division=0)[0],
        "confusion_matrix": cm,
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Test model accuracy")
    parser.add_argument("--model", type=str, default="models/resnet18_best.pt", help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = build_model()
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Load dataset
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SafeImageFolder(root=args.data_dir, transform=val_transform)
    print(f"Dataset: {len(dataset)} valid images")
    print(f"Classes: {dataset.class_to_idx}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(model, loader, device, args.threshold)

    # Print results
    cm = metrics["confusion_matrix"]
    print("\n" + "="*60)
    print("  MODEL ACCURACY TEST RESULTS")
    print("="*60)
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (AI):  {metrics['precision']:.4f}")
    print(f"  Recall (AI):     {metrics['recall']:.4f}")
    print(f"  F1-Score (AI):   {metrics['f1_score']:.4f}")
    print("-"*60)
    print(f"  Precision (Real): {metrics['precision_real']:.4f}")
    print(f"  Recall (Real):    {metrics['recall_real']:.4f}")
    print("-"*60)
    print(f"  Confusion Matrix:")
    print(f"                    Pred:Real  Pred:AI")
    print(f"  Actual:Real        {cm[0,0]:6d}     {cm[0,1]:6d}   (FP={cm[0,1]})")
    print(f"  Actual:AI          {cm[1,0]:6d}     {cm[1,1]:6d}   (FN={cm[1,0]})")
    print("="*60)
    print(f"\n  False Positives (Real→AI): {metrics['false_positives']}")
    print(f"  False Negatives (AI→Real): {metrics['false_negatives']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
