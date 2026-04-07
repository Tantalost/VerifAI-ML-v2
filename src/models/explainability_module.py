import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import logging
from sklearn.decomposition import PCA
from ultralytics import YOLO
import os
from typing import Dict, Any, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv8Explainer:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the YOLOv8 explainer with a trained model.
        
        Args:
            model_path: Path to the trained YOLOv8 classification model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.class_names = ['real', 'ai_generated']
        
        # Get the underlying PyTorch model for gradient computation
        self.pytorch_model = self.model.model
        
        logger.info(f"Loaded model from {model_path} on {self.device}")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for YOLOv8 inference.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_image_array)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # YOLOv8 preprocessing
        image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_array = np.array(image_resized) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_image
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from different layers of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of layer names to feature maps
        """
        feature_maps = {}
        
        # Hook function to capture feature maps
        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        # Register hooks for different layers
        hooks = []
        
        # Use named_modules to find all Conv2d layers in the model
        for name, module in self.pytorch_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.pytorch_model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return feature_maps
    
    def compute_eigencam(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute EigenCAM using principal component analysis on feature maps.
        
        Args:
            feature_maps: Dictionary of feature maps
            
        Returns:
            EigenCAM heatmap
        """
        # Select the most informative feature maps (usually the deepest ones)
        # We'll use the last few layers for better semantic information
        selected_features = []
        
        # Sort layer names to get the deeper layers (higher numbers)
        sorted_layers = sorted(feature_maps.items(), key=lambda x: x[0])
        
        # Select the last 10 layers or all if less than 10
        for name, features in sorted_layers[-10:]:
            # Apply adaptive average pooling to standardize spatial dimensions
            spatial_size = features.shape[2:]  # height, width
            if spatial_size != (7, 7):  # Standardize to 7x7
                pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (7, 7))
            else:
                pooled_features = features
            
            # Global average pooling across channels to get single value per spatial location
            # [batch, channels, height, width] -> [batch, height, width]
            gap_features = torch.mean(pooled_features, dim=1)
            selected_features.append(gap_features)
        
        if not selected_features:
            raise ValueError("No feature maps found for EigenCAM computation")
        
        # Stack the features along the channel dimension
        # Each tensor is [batch, height, width], stack to [batch, num_layers, height, width]
        stacked_features = torch.stack(selected_features, dim=1)
        
        # Reshape for PCA: [batch, num_layers, height, width] -> [batch, height*width, num_layers]
        batch_size, num_layers, height, width = stacked_features.shape
        features_2d = stacked_features.view(batch_size, height * width, num_layers).squeeze(0).cpu().numpy()
        
        # Center the data
        features_centered = features_2d - np.mean(features_2d, axis=0, keepdims=True)
        
        # Compute covariance matrix
        cov_matrix = np.cov(features_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get the principal component (largest eigenvalue)
        principal_component = eigenvectors[:, -1]
        
        # Project features onto principal component
        cam_weights = np.dot(features_centered, principal_component)
        
        # Reshape back to spatial dimensions
        # We need to determine the original spatial dimensions
        # Let's use the first feature map's spatial dimensions
        first_features = list(feature_maps.values())[0]
        h, w = first_features.shape[2], first_features.shape[3]
        
        # Reshape CAM weights
        spatial_points = cam_weights.shape[0]
        # Approximate spatial dimensions (this is a simplification)
        # In practice, you might need to track the exact spatial dimensions
        cam_reshaped = cam_weights.reshape(int(np.sqrt(spatial_points)), int(np.sqrt(spatial_points)))
        
        # Resize to match original feature map size
        cam_resized = cv2.resize(cam_reshaped, (w, h))
        
        # Normalize to [0, 1]
        cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
        return torch.from_numpy(cam_normalized).float().to(self.device)
    
    def compute_gradcam(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute Grad-CAM for a specific target class.
        
        Args:
            x: Input tensor
            target_class: Target class index (0 for real, 1 for ai_generated)
            
        Returns:
            Grad-CAM heatmap
        """
        # Get the target layer (usually the last convolutional layer)
        target_layer = None
        for layer in reversed(list(self.pytorch_model.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break
        
        if target_layer is None:
            raise ValueError("No convolutional layer found for Grad-CAM")
        
        # Hook to capture gradients and activations
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Enable gradients for Grad-CAM computation
            x.requires_grad_(True)
            
            # Forward pass
            output = self.pytorch_model(x)
            
            # Handle YOLOv8 output format (may return tuple)
            if isinstance(output, tuple):
                output = output[0] if len(output) > 0 else output
            
            # Zero gradients
            self.pytorch_model.zero_grad()
            
            # Create one-hot encoding for target class
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            
            # Backward pass
            output.backward(gradient=one_hot)
            
            # Compute Grad-CAM
            if activations is not None and gradients is not None:
                # Global average pooling of gradients
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * activations, dim=1)
                
                # ReLU to keep only positive influences
                cam = F.relu(cam)
                
                # Normalize to [0, 1]
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                return cam.squeeze(0)
            else:
                raise ValueError("Could not compute activations or gradients")
        
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
    
    def generate_heatmap_overlay(self, original_image: np.ndarray, heatmap: torch.Tensor, 
                               colormap: str = 'jet', alpha: float = 0.4) -> np.ndarray:
        """
        Generate heatmap overlay on the original image.
        
        Args:
            original_image: Original image array
            heatmap: CAM heatmap tensor
            colormap: Matplotlib colormap name
            alpha: Transparency of overlay
            
        Returns:
            Overlay image array
        """
        # Convert heatmap to numpy and resize to original image size
        heatmap_np = heatmap.cpu().numpy()
        heatmap_resized = cv2.resize(heatmap_np, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        colormap_fn = cm.get_cmap(colormap)
        heatmap_colored = colormap_fn(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def detect_ai_regions(self, heatmap: torch.Tensor, original_image: np.ndarray, 
                         confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect AI-generated regions using heatmap analysis with YOLOv8-style bounding boxes.
        
        Args:
            heatmap: The explainability heatmap
            original_image: Original image as numpy array
            confidence_threshold: Threshold for region detection
            
        Returns:
            List of detected regions with bounding boxes and labels
        """
        try:
            # Convert heatmap to numpy
            if isinstance(heatmap, torch.Tensor):
                heatmap_np = heatmap.cpu().numpy()
            else:
                heatmap_np = heatmap
            
            # Ensure heatmap is in [0, 1] range
            heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
            
            # Resize heatmap to match original image
            original_h, original_w = original_image.shape[:2]
            heatmap_resized = cv2.resize(heatmap_np, (original_w, original_h))
            
            # Apply threshold to create binary mask
            binary_mask = (heatmap_resized > confidence_threshold).astype(np.uint8)
            
            # Find contours (potential regions)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_regions = []
            
            for i, contour in enumerate(contours):
                # Calculate contour area and filter small regions
                area = cv2.contourArea(contour)
                if area < (original_w * original_h * 0.01):  # Filter regions smaller than 1% of image
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate region confidence based on heatmap intensity
                region_heatmap = heatmap_resized[y:y+h, x:x+w]
                region_confidence = np.mean(region_heatmap)
                
                # Determine label based on region characteristics
                label = self._classify_region_type(region_heatmap, region_confidence)
                
                detected_regions.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(region_confidence),
                    'label': label,
                    'area': float(area),
                    'region_id': i
                })
            
            # Sort regions by confidence
            detected_regions.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Detected {len(detected_regions)} AI-generated regions")
            return detected_regions
            
        except Exception as e:
            logger.error(f"Error in region detection: {str(e)}")
            return []
    
    def _classify_region_type(self, region_heatmap: np.ndarray, confidence: float) -> str:
        """
        Classify the type of AI-generated region based on heatmap patterns.
        
        Args:
            region_heatmap: Heatmap values for the region
            confidence: Average confidence score
            
        Returns:
            String label for the region type
        """
        # Calculate region characteristics
        std_dev = np.std(region_heatmap)
        mean_val = np.mean(region_heatmap)
        
        # Classify based on patterns
        if confidence > 0.7:
            if std_dev > 0.2:
                return "AI Pattern"
            else:
                return "Anomaly"
        elif confidence > 0.5:
            if std_dev > 0.15:
                return "Artifact"
            else:
                return "Suspicious"
        else:
            return "Uncertain"
    
    def draw_bounding_boxes(self, original_image: np.ndarray, 
                           detected_regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw YOLOv8-style bounding boxes on the original image.
        
        Args:
            original_image: Original image as numpy array
            detected_regions: List of detected regions
            
        Returns:
            Image with bounding boxes drawn
        """
        try:
            # Create a copy of the original image
            image_with_boxes = original_image.copy()
            
            # Define colors for different labels
            colors = {
                "AI Pattern": (255, 0, 0),      # Red
                "Anomaly": (0, 255, 0),         # Green
                "Artifact": (0, 0, 255),        # Blue
                "Suspicious": (255, 255, 0),    # Yellow
                "Uncertain": (128, 128, 128)    # Gray
            }
            
            for region in detected_regions:
                bbox = region['bbox']
                x, y, w, h = bbox
                confidence = region['confidence']
                label = region['label']
                color = colors.get(label, (255, 0, 0))
                
                # Draw bounding box
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
                
                # Create label background
                label_text = f"{label}: {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(image_with_boxes, (x, y - label_height - 10), 
                            (x + label_width, y), color, -1)
                
                # Draw label text
                cv2.putText(image_with_boxes, label_text, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return image_with_boxes
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {str(e)}")
            return original_image
    
    def explain_image(self, image_path: str, method: str = 'eigencam', 
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate explanation for an image using the specified method.
        
        Args:
            image_path: Path to the input image
            method: Explanation method ('eigencam' or 'gradcam')
            save_path: Path to save the explanation visualization
            
        Returns:
            Dictionary containing prediction, confidence, and visualization
        """
        logger.info(f"Explaining image {image_path} using {method}")
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.pytorch_model(image_tensor)
            
            # Handle YOLOv8 output format (may return tuple)
            if isinstance(outputs, tuple):
                outputs = outputs[0] if len(outputs) > 0 else outputs
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class_idx = predicted_class.item()
            confidence_score = confidence.item()
            predicted_label = self.class_names[predicted_class_idx]
        
        logger.info(f"Prediction: {predicted_label} with confidence {confidence_score:.4f}")
        
        # Generate heatmap
        if method.lower() == 'eigencam':
            feature_maps = self.get_feature_maps(image_tensor)
            heatmap = self.compute_eigencam(feature_maps)
        elif method.lower() == 'gradcam':
            heatmap = self.compute_gradcam(image_tensor, predicted_class_idx)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'eigencam' or 'gradcam'")
        
        # Generate overlay
        overlay = self.generate_heatmap_overlay(original_image, heatmap)
        
        # Detect AI regions and draw bounding boxes
        detected_regions = self.detect_ai_regions(heatmap, original_image)
        image_with_boxes = self.draw_bounding_boxes(original_image, detected_regions)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap.cpu().numpy(), cmap='jet')
        axes[1].set_title(f'{method.upper()} Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Prediction: {predicted_label}\nConfidence: {confidence_score:.3f}')
        axes[2].axis('off')
        
        # Bounding boxes
        axes[3].imshow(image_with_boxes)
        axes[3].set_title(f'Detected Regions: {len(detected_regions)}')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation saved to {save_path}")
        
        result = {
            'predicted_class': predicted_label,
            'confidence': confidence_score,
            'class_probabilities': {
                'real': probabilities[0, 0].item(),
                'ai_generated': probabilities[0, 1].item()
            },
            'heatmap': heatmap.cpu().numpy(),
            'overlay': overlay,
            'image_with_boxes': image_with_boxes,
            'original_image': original_image,
            'detected_regions': detected_regions,
            'visualization': fig
        }
        
        return result
    
    def batch_explain(self, image_paths: list, method: str = 'eigencam', 
                     output_dir: str = 'explanations') -> list:
        """
        Generate explanations for a batch of images.
        
        Args:
            image_paths: List of image paths
            method: Explanation method
            output_dir: Directory to save explanations
            
        Returns:
            List of explanation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                save_path = output_path / f"explanation_{i+1}.png"
                result = self.explain_image(image_path, method, str(save_path))
                result['image_path'] = image_path
                result['save_path'] = str(save_path)
                results.append(result)
                
                logger.info(f"Processed {i+1}/{len(image_paths)}: {image_path}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

def main():
    """Example usage of the explainability module"""
    # Example usage (uncomment to test)
    # explainer = YOLOv8Explainer('path/to/your/trained/model.pt')
    
    # # Single image explanation
    # result = explainer.explain_image('path/to/image.jpg', method='eigencam')
    # plt.show()
    
    # # Batch explanation
    # image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = explainer.batch_explain(image_paths, method='gradcam')
    
    print("🔍 YOLOv8 Explainability Module loaded successfully!")
    print("Usage:")
    print("  explainer = YOLOv8Explainer('path/to/model.pt')")
    print("  result = explainer.explain_image('image.jpg', method='eigencam')")

if __name__ == "__main__":
    main()
