"""
Google Colab GPU Training Script for VerifAI-ML
================================================
Paste this into a Colab notebook cell and run.

BEFORE RUNNING:
1. Zip your dataset folder: dataset/real/ + dataset/ai/
2. Upload the zip to Google Drive root as "verifai_dataset.zip"
3. In Colab: Runtime > Change runtime type > GPU (T4)
4. Paste and run this script

Expected time: ~15-30 minutes for 25 epochs on T4 GPU
"""

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Check GPU
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Step 3: Install dependencies
!pip install -q torch torchvision scikit-learn tqdm

# Step 4: Unzip dataset from Drive
import os
import zipfile

zip_path = "/content/drive/MyDrive/verifai_dataset.zip"
extract_path = "/content/VerifAI-ML"

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Verify dataset
real_count = len([f for f in os.listdir(f"{extract_path}/dataset/real") if f.endswith('.jpg')])
ai_count = len([f for f in os.listdir(f"{extract_path}/dataset/ai") if f.endswith('.jpg')])
print(f"Dataset: {real_count} real + {ai_count} AI = {real_count + ai_count} total images")

# Step 5: Create train.py on Colab
train_py = """
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


def get_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_model(model_name, num_classes=2, dropout=0.5):
    model = models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    return running_loss / total, correct / total


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
    import sys
    # Parse simple args
    args = {
        'data_dir': '/content/VerifAI-ML/dataset',
        'output_dir': '/content/drive/MyDrive/VerifAI-ML/models',
        'epochs': 25,
        'batch_size': 64,
        'lr': 1e-4,
        'img_size': 224,
        'train_split': 0.8,
        'real_weight': 1.5,
        'threshold': 0.5,
        'seed': 42,
        'dropout': 0.5,
        'weight_decay': 1e-5,
        'save_every': 5,
    }

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args['output_dir'], exist_ok=True)

    train_tf = get_transforms(args['img_size'], True)
    val_tf = get_transforms(args['img_size'], False)

    full_ds = SafeImageFolder(root=args['data_dir'], transform=train_tf)
    class_to_idx = full_ds.class_to_idx
    print(f"Classes: {class_to_idx}, Total valid: {len(full_ds)}")

    n_train = int(len(full_ds) * args['train_split'])
    n_val = len(full_ds) - n_train
    indices = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(args['seed'])).tolist()
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds = SafeImageFolder(root=args['data_dir'], transform=val_tf)
    val_ds = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True)

    model = build_model('resnet18', dropout=args['dropout']).to(device)

    real_idx = class_to_idx.get("real", 0)
    weights = torch.tensor([args['real_weight'] if real_idx == 0 else 1.0,
                            args['real_weight'] if real_idx == 1 else 1.0]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print(f"Training: {args['epochs']} epochs, bs={args['batch_size']}, lr={args['lr']}")
    print("="*60)

    best_f1 = 0.0
    for epoch in range(1, args['epochs'] + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device, args['threshold'])
        scheduler.step(metrics['f1_score'])
        elapsed = time.time() - start

        print(f"Epoch {epoch:02d}/{args['epochs']} | {elapsed:.1f}s | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        if epoch % args['save_every'] == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics},
                       f"{args['output_dir']}/epoch{epoch}.pt")

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics},
                       f"{args['output_dir']}/best.pt")

    # Final print
    best_ckpt = torch.load(f"{args['output_dir']}/best.pt", map_location=device)
    model.load_state_dict(best_ckpt['state_dict'])
    final = evaluate(model, val_loader, device, args['threshold'])
    cm = final['confusion_matrix']
    print("\\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Accuracy:        {final['accuracy']:.4f}")
    print(f"Precision (AI):  {final['precision']:.4f}")
    print(f"Recall (AI):     {final['recall']:.4f}")
    print(f"F1-Score (AI):   {final['f1_score']:.4f}")
    print(f"Precision (Real): {final['precision_real']:.4f}")
    print(f"Recall (Real):    {final['recall_real']:.4f}")
    print(f"Confusion Matrix: Real->AI(FP)={cm[0,1]}, AI->Real(FN)={cm[1,0]}")
    print("="*60)
    print(f"Models saved to: {args['output_dir']}")

if __name__ == "__main__":
    main()
"""

with open("/content/train_colab.py", "w") as f:
    f.write(train_py)

print("train_colab.py created successfully!")

# Step 6: Run training
print("\\n🏋️ Starting GPU training...")
!python /content/train_colab.py

print("\\n✅ Training complete! Models saved to Google Drive.")
print("📁 Check: /content/drive/MyDrive/VerifAI-ML/models/")
