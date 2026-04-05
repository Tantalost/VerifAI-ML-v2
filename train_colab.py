"""
Google Colab Training Script for VerifAI-ML
Run this in Google Colab to train the model without local storage issues
"""

# Install dependencies
!pip install torch torchvision ultralytics numpy pandas Pillow opencv-python scikit-learn matplotlib plotly seaborn PyYAML tqdm streamlit

# Mount Google Drive for storage
from google.colab import drive
drive.mount('/content/drive')

# Download datasets (reduced size for Colab)
import os
from datasets import load_dataset
from tqdm import tqdm
import requests
import zipfile

# Create directories
!mkdir -p dataset/real dataset/ai_generated

print("Downloading AI-generated images (1000 images)...")
dataset = load_dataset("ash12321/sdxl-generated-10k", split="train[:1000]")
for i, item in enumerate(tqdm(dataset)):
    image = item['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(f"dataset/ai_generated/ai_sdxl_{i}.jpg", "JPEG", quality=95)

print("Downloading real images (1000 images)...")
# Use a smaller real image dataset for Colab
real_dataset = load_dataset("cifar10", split="test[:1000]")
for i, item in enumerate(tqdm(real_dataset)):
    image = item['img']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(f"dataset/real/real_{i}.jpg", "JPEG", quality=95)

# Run data preparation
print("Preparing dataset...")
!python src/data/data_preparation.py

# Train the model
print("Starting training...")
!python src/models/training_pipeline.py

# Save model to Google Drive
!cp best.pt /content/drive/MyDrive/verifai_model.pt
print("Model saved to Google Drive!")
