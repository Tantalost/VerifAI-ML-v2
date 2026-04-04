#!/usr/bin/env python3
"""
VerifAI-ML Complete Setup Script
Automated setup for the entire AI image detection system
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install all required packages"""
    print("📦 Installing requirements...")
    
    # Install base requirements
    requirements_files = [
        "requirements_phase1.txt",
        "requirements_phase2.txt", 
        "requirements_phase3.txt",
        "requirements_phase4.txt"
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"  Installing {req_file}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file])
    
    print("✅ Requirements installed!")

def check_dataset():
    """Check if dataset exists and has images"""
    print("📁 Checking dataset...")
    
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        print("Please create 'dataset/' with 'real/' and 'ai_generated/' subdirectories")
        return False
    
    real_path = dataset_path / "real"
    ai_path = dataset_path / "ai_generated"
    
    real_count = len(list(real_path.glob("*.jpg"))) + len(list(real_path.glob("*.png")))
    ai_count = len(list(ai_path.glob("*.jpg"))) + len(list(ai_path.glob("*.png")))
    
    print(f"  Real images: {real_count}")
    print(f"  AI-generated images: {ai_count}")
    
    if real_count == 0 or ai_count == 0:
        print("❌ Dataset is empty or incomplete!")
        return False
    
    print("✅ Dataset looks good!")
    return True

def prepare_data():
    """Run data preparation"""
    print("🔄 Preparing data...")
    
    if not os.path.exists("yolo_dataset"):
        subprocess.run([sys.executable, "data_preparation.py"])
        print("✅ Data preparation completed!")
    else:
        print("✅ Data already prepared!")

def train_model():
    """Train the model"""
    print("🏋️ Training model...")
    
    # Check if model already exists
    model_paths = [
        "runs/classify/yolov8_classification_*/weights/best.pt",
        "best.pt"
    ]
    
    model_exists = False
    for pattern in model_paths:
        if "*" in pattern:
            from glob import glob
            if glob(pattern):
                model_exists = True
                break
        else:
            if os.path.exists(pattern):
                model_exists = True
                break
    
    if not model_exists:
        print("  Starting training (this may take a while)...")
        subprocess.run([sys.executable, "training_pipeline.py"])
        print("✅ Model training completed!")
    else:
        print("✅ Model already trained!")

def main():
    """Complete setup process"""
    print("🚀 VerifAI-ML Complete Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    print()
    
    # Step 2: Check dataset
    if not check_dataset():
        print("\n❌ Setup failed: Dataset issues")
        return
    
    print()
    
    # Step 3: Prepare data
    prepare_data()
    print()
    
    # Step 4: Train model
    train_model()
    print()
    
    # Step 5: Ready to launch
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("🌐 Launch the app with:")
    print("  streamlit run app.py")
    print("  or")
    print("  python run_app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
