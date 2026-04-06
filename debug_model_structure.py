#!/usr/bin/env python3
"""
Debug script to understand YOLOv8 model structure
"""
import torch
from ultralytics import YOLO

def debug_model_structure():
    """Print YOLOv8 model structure to understand layer organization"""
    
    # Load the model
    model = YOLO("best.pt")
    pytorch_model = model.model
    
    print("🔍 YOLOv8 Model Structure:")
    print(f"Model type: {type(pytorch_model)}")
    print(f"Total layers: {len(list(pytorch_model.modules()))}")
    
    print("\n📋 Model components:")
    for i, (name, module) in enumerate(pytorch_model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            print(f"  {i}: {name} - Conv2d ({module.in_channels} -> {module.out_channels})")
    
    print(f"\n🏗️  Model architecture:")
    print(f"  Model: {pytorch_model}")
    
    # Try to understand the structure better
    if hasattr(pytorch_model, 'model'):
        print(f"\n📦 Sub-model structure:")
        for i, layer in enumerate(pytorch_model.model):
            print(f"  {i}: {type(layer).__name__}")
            if isinstance(layer, torch.nn.Conv2d):
                print(f"      Conv2d: {layer.in_channels} -> {layer.out_channels}")

if __name__ == "__main__":
    debug_model_structure()
