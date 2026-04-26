#!/usr/bin/env python3
"""
Quick test script for the trained best.pt model
"""
import sys
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
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

def evaluate_model():
    """Evaluate model accuracy and report false positive/negative behavior"""
    model_path = "best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found!")
        return
    
    print("🔄 Evaluating model...")
    
    model = YOLO(model_path)
    metrics = model.val()
    
    print("📊 Evaluation Results:")
    print(metrics)
    
    results = getattr(metrics, 'results_dict', {}) or {}
    top1 = results.get('metrics/accuracy_top1', getattr(metrics, 'top1', None))
    top5 = results.get('metrics/accuracy_top5', getattr(metrics, 'top5', None))
    if top1 is not None:
        print(f"✅ Top-1 Accuracy: {top1 * 100:.2f}%")
    if top5 is not None:
        print(f"✅ Top-5 Accuracy: {top5 * 100:.2f}%")
    
    cm = getattr(metrics, 'confusion_matrix', None)
    if cm is not None:
        print("\n🔢 Confusion matrix (rows=true label, cols=predicted label):")
        try:
            matrix = np.asarray(getattr(cm, 'matrix', cm))
        except Exception:
            matrix = np.asarray(cm)
        print(matrix)

        if matrix.ndim == 2 and matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
            if matrix.shape != (2, 2) and np.all(matrix[2:, :] == 0) and np.all(matrix[:, 2:] == 0):
                print("ℹ️ Extra zero-padding detected in confusion matrix; using first 2x2 block for binary metrics.")
                matrix = matrix[:2, :2]

            if matrix.shape == (2, 2):
                tn, fp, fn, tp = matrix.ravel()
                fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
                fn_rate = fn / (fn + tp) if (fn + tp) else 0.0
                print(f"✅ False positive rate: {fp_rate * 100:.2f}% ({fp}/{fp + tn if fp + tn else 0})")
                print(f"✅ False negative rate: {fn_rate * 100:.2f}% ({fn}/{fn + tp if fn + tp else 0})")
            else:
                print("⚠️ Confusion matrix is larger than 2x2 and cannot be reduced to a binary false positive/negative summary automatically.")
        else:
            print("⚠️ Confusion matrix is not valid for binary false positive/negative computation.")
    else:
        print("⚠️ No confusion matrix available in metrics object.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_with_image(sys.argv[1])
    else:
        test_model()
        evaluate_model()