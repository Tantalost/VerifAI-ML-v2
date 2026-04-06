#!/usr/bin/env python3
"""
Quick test script for the trained best.pt model
"""
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.explainability_module import YOLOv8Explainer

def test_model():
    """Test the trained model with explainability"""
    
    # Check if best.pt exists
    model_path = "best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found!")
        return False
    
    print(f"🔄 Loading model from {model_path}...")
    
    try:
        # Initialize explainer
        explainer = YOLOv8Explainer(model_path)
        print("✅ Model loaded successfully!")
        
        # Test with a sample image (you'll need to provide one)
        print("📝 To test with an image, run:")
        print("   python test_model.py path/to/your/image.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_with_image(image_path):
    """Test model with a specific image"""
    if not Path(image_path).exists():
        print(f"❌ Image {image_path} not found!")
        return False
    
    try:
        explainer = YOLOv8Explainer("best.pt")
        
        # Get prediction
        result = explainer.explain_image(image_path, method='eigencam')
        
        print(f"🎯 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"🔍 Explanation saved to: {result.get('output_path', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided image
        test_with_image(sys.argv[1])
    else:
        # Just test model loading
        test_model()
