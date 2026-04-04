import os
import torch
from pathlib import Path
import logging
from datetime import datetime
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv8Trainer:
    def __init__(self, dataset_path="yolo_dataset", model_name="yolov8m-cls.pt"):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.model = None
        self.results = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Using device: {self.device}")
        
    def create_dataset_yaml(self):
        """Create dataset.yaml configuration file for YOLOv8"""
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': 2,  # Number of classes
            'names': ['real', 'ai_generated']
        }
        
        yaml_path = self.dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Created dataset configuration at {yaml_path}")
        return yaml_path
    
    def load_model(self):
        """Load pre-trained YOLOv8 classification model"""
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"Loaded pre-trained model: {self.model_name}")
            
            # Print model info
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_training_params(self):
        """Setup optimal training parameters"""
        # Calculate optimal batch size based on GPU memory
        if self.device == 'cuda':
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            if gpu_memory >= 12:
                batch_size = 32
            elif gpu_memory >= 8:
                batch_size = 16
            elif gpu_memory >= 6:
                batch_size = 8
            else:
                batch_size = 4
        else:
            batch_size = 8  # Conservative for CPU
        
        logger.info(f"Using batch size: {batch_size}")
        
        training_params = {
            'data': str(self.dataset_path / 'dataset.yaml'),
            'epochs': 100,
            'batch': batch_size,
            'imgsz': 224,  # Standard image size for classification
            'device': self.device,
            'project': 'VerifAI-ML',
            'name': f'yolov8_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': False,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,  # Initial learning rate
            'lrf': 0.01,   # Final learning rate fraction
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,     # Box loss weight
            'cls': 0.5,     # Classification loss weight
            'dfl': 1.5,     # Distribution focal loss weight
            'pose': 12.0,   # Pose loss weight
            'kobj': 1.0,    # Keypoint object loss weight
            'label_smoothing': 0.0,
            'nbs': 64,      # Nominal batch size
            'hsv_h': 0.015, # HSV hue augmentation
            'hsv_s': 0.7,   # HSV saturation augmentation
            'hsv_v': 0.4,   # HSV value augmentation
            'degrees': 0.0, # Rotation augmentation
            'translate': 0.1, # Translation augmentation
            'scale': 0.5,   # Scale augmentation
            'shear': 0.0,   # Shear augmentation
            'perspective': 0.0, # Perspective augmentation
            'flipud': 0.0,  # Vertical flip augmentation
            'fliplr': 0.5,  # Horizontal flip augmentation
            'mosaic': 0.0,  # Mosaic augmentation
            'mixup': 0.0,   # Mixup augmentation
            'copy_paste': 0.0, # Copy-paste augmentation
            'auto_augment': 'randaugment', # Auto augmentation
            'erasing': 0.4, # Random erasing
            'crop_fraction': 1.0, # Crop fraction
            'patience': 15, # Early stopping patience
            'save': True,
            'save_period': 10, # Save every 10 epochs
            'cache': True,   # Cache images for faster training
            'workers': 8,    # Data loading workers
            'close_mosaic': 10, # Disable mosaic augmentation in final epochs
            'resume': False,
            'amp': True,     # Automatic Mixed Precision
            'fraction': 1.0, # Dataset fraction to train on
            'profile': False,
            'freeze': None,  # Layers to freeze
            'multi_scale': False,
            'single_cls': False,
            'seed': 42,      # Random seed for reproducibility
            'deterministic': True,
            'plots': True,   # Generate training plots
            'val': True,     # Validate during training
            'split': 'val',  # Validation split
            'save_json': False,
            'save_hybrid': False,
            'conf': None,    # Confidence threshold
            'iou': 0.7,      # IoU threshold for NMS
            'max_det': 300,  # Maximum detections per image
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'project_root': str(self.dataset_path.parent)
        }
        
        return training_params
    
    def train(self):
        """Train the YOLOv8 classification model"""
        if self.model is None:
            self.load_model()
        
        # Create dataset configuration
        yaml_path = self.create_dataset_yaml()
        
        # Get training parameters
        params = self.setup_training_params()
        
        logger.info("Starting YOLOv8 training...")
        logger.info(f"Training parameters: {params}")
        
        try:
            # Start training
            self.results = self.model.train(**params)
            
            logger.info("Training completed successfully!")
            
            # Save the best model
            best_model_path = self.results.save_dir / 'weights' / 'best.pt'
            logger.info(f"Best model saved at: {best_model_path}")
            
            return self.results, best_model_path
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate_model(self, model_path=None):
        """Evaluate the trained model on test set"""
        if model_path:
            model = YOLO(model_path)
        elif self.results:
            model = self.results.model
        else:
            logger.error("No trained model available for evaluation")
            return None
        
        logger.info("Evaluating model on test set...")
        
        try:
            # Evaluate on test set
            test_results = model.val(
                data=str(self.dataset_path / 'dataset.yaml'),
                split='test',
                device=self.device,
                plots=True,
                save_json=True
            )
            
            logger.info("Test evaluation completed!")
            
            # Print metrics
            metrics = test_results.results_dict
            print("\n" + "="*50)
            print("TEST SET EVALUATION METRICS")
            print("="*50)
            print(f"Accuracy: {metrics.get('metrics/accuracy_top1', 0):.4f}")
            print(f"Top-5 Accuracy: {metrics.get('metrics/accuracy_top5', 0):.4f}")
            print(f"Precision: {metrics.get('metrics/precision', 0):.4f}")
            print(f"Recall: {metrics.get('metrics/recall', 0):.4f}")
            print(f"F1-Score: {metrics.get('metrics/f1', 0):.4f}")
            print("="*50)
            
            return test_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return None
    
    def plot_training_curves(self):
        """Plot training curves if available"""
        if not self.results:
            logger.warning("No training results available for plotting")
            return
        
        try:
            # Load training history
            results_csv = self.results.save_dir / 'results.csv'
            
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                
                # Create subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('YOLOv8 Training Curves', fontsize=16)
                
                # Loss curves
                axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss')
                axes[0, 0].plot(df['epoch'], df['val/loss'], label='Val Loss')
                axes[0, 0].set_title('Training and Validation Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                # Accuracy curves
                if 'metrics/accuracy_top1' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'], label='Top-1 Accuracy', color='green')
                    axes[0, 1].set_title('Top-1 Accuracy')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Accuracy')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                
                # Precision/Recall
                if 'metrics/precision' in df.columns and 'metrics/recall' in df.columns:
                    axes[1, 0].plot(df['epoch'], df['metrics/precision'], label='Precision', color='orange')
                    axes[1, 0].plot(df['epoch'], df['metrics/recall'], label='Recall', color='purple')
                    axes[1, 0].set_title('Precision and Recall')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Score')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                
                # Learning rate
                if 'lr/pg0' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='red')
                    axes[1, 1].set_title('Learning Rate Schedule')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Learning Rate')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                    axes[1, 1].set_yscale('log')
                
                plt.tight_layout()
                
                # Save plot
                plot_path = self.results.save_dir / 'training_curves.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training curves saved to: {plot_path}")
                
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training curves: {str(e)}")

def main():
    """Main function to run the training pipeline"""
    # Initialize trainer
    trainer = YOLOv8Trainer()
    
    # Check if dataset exists
    if not trainer.dataset_path.exists():
        logger.error(f"Dataset path {trainer.dataset_path} does not exist!")
        logger.error("Please run data_preparation.py first to create the dataset.")
        return
    
    # Load pre-trained model
    trainer.load_model()
    
    # Train the model
    results, best_model_path = trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate_model(best_model_path)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"📁 Results saved in: {results.save_dir}")
    print(f"🏆 Best model: {best_model_path}")

if __name__ == "__main__":
    main()
