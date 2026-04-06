#!/usr/bin/env python3
"""
Debug script to test feature map extraction
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.explainability_module import YOLOv8Explainer

def test_feature_maps():
    """Test feature map extraction with a dummy tensor"""
    
    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("🔄 Testing feature map extraction...")
    
    try:
        # Initialize explainer
        explainer = YOLOv8Explainer("best.pt")
        
        # Extract feature maps
        feature_maps = explainer.get_feature_maps(dummy_input)
        
        print(f"✅ Found {len(feature_maps)} feature maps:")
        for name, features in feature_maps.items():
            print(f"  - {name}: {features.shape}")
        
        if len(feature_maps) == 0:
            print("❌ No feature maps found!")
        else:
            print("✅ Feature map extraction successful!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_maps()
