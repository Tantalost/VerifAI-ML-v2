import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, dataset_path="dataset", output_path="yolo_dataset"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.classes = ["real", "ai_generated"]
        
    def create_yolo_structure(self):
        """Create YOLOv8 classification directory structure"""
        # Create main directories
        splits = ["train", "val", "test"]
        
        for split in splits:
            split_path = self.output_path / split
            split_path.mkdir(parents=True, exist_ok=True)
            
            # Create class subdirectories
            for class_name in self.classes:
                class_path = split_path / class_name
                class_path.mkdir(parents=True, exist_ok=True)
                
        logger.info(f"Created YOLOv8 directory structure in {self.output_path}")
    
    def aggressive_augmentation(self, image_path, save_dir, base_name):
        """Apply aggressive data augmentation to prevent overfitting"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            augmented_images = []
            
            # 1. Original image
            augmented_images.append((img.copy(), f"{base_name}_original.jpg"))
            
            # 2. Horizontal flip
            if random.random() > 0.3:  # 70% chance
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                augmented_images.append((flipped, f"{base_name}_flip.jpg"))
            
            # 3. Random rotation (-15 to +15 degrees)
            if random.random() > 0.4:  # 60% chance
                angle = random.uniform(-15, 15)
                rotated = img.rotate(angle, expand=False, fillcolor='white')
                augmented_images.append((rotated, f"{base_name}_rot{angle:.0f}.jpg"))
            
            # 4. Brightness adjustment
            if random.random() > 0.5:  # 50% chance
                enhancer = ImageEnhance.Brightness(img)
                factor = random.uniform(0.7, 1.3)
                bright = enhancer.enhance(factor)
                augmented_images.append((bright, f"{base_name}_bright{factor:.1f}.jpg"))
            
            # 5. Contrast adjustment
            if random.random() > 0.5:  # 50% chance
                enhancer = ImageEnhance.Contrast(img)
                factor = random.uniform(0.7, 1.3)
                contrast = enhancer.enhance(factor)
                augmented_images.append((contrast, f"{base_name}_contrast{factor:.1f}.jpg"))
            
            # 6. Gaussian blur (slight)
            if random.random() > 0.7:  # 30% chance
                blurred = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                augmented_images.append((blurred, f"{base_name}_blur.jpg"))
            
            # 7. Noise injection
            if random.random() > 0.6:  # 40% chance
                img_array = np.array(img)
                noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
                noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(noisy)
                augmented_images.append((noisy_img, f"{base_name}_noise.jpg"))
            
            # 8. Saturation adjustment
            if random.random() > 0.6:  # 40% chance
                enhancer = ImageEnhance.Color(img)
                factor = random.uniform(0.7, 1.3)
                saturated = enhancer.enhance(factor)
                augmented_images.append((saturated, f"{base_name}_sat{factor:.1f}.jpg"))
            
            # 9. Crop and resize (simulating different perspectives)
            if random.random() > 0.7:  # 30% chance
                width, height = img.size
                # Crop 80-95% of the image
                crop_factor = random.uniform(0.8, 0.95)
                left = random.uniform(0, width * (1 - crop_factor))
                top = random.uniform(0, height * (1 - crop_factor))
                right = left + width * crop_factor
                bottom = top + height * crop_factor
                
                cropped = img.crop((left, top, right, bottom))
                resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
                augmented_images.append((resized, f"{base_name}_crop.jpg"))
            
            # Save augmented images
            saved_paths = []
            for aug_img, filename in augmented_images:
                save_path = save_dir / filename
                aug_img.save(save_path, "JPEG", quality=95)
                saved_paths.append(save_path)
            
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {str(e)}")
            return []
    
    def split_and_augment_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, augment_factor=3):
        """Split data and apply augmentation"""
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                logger.error(f"Class directory {class_path} does not exist!")
                continue
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(class_path.glob(ext))
                image_files.extend(class_path.glob(ext.upper()))
            
            logger.info(f"Found {len(image_files)} images in {class_name}")
            
            if len(image_files) == 0:
                logger.warning(f"No images found in {class_path}")
                continue
            
            # Shuffle files
            random.shuffle(image_files)
            
            # Calculate split indices
            n_files = len(image_files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            logger.info(f"Split {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
            
            # Process each split
            splits_data = [
                (train_files, "train", True),   # Apply augmentation to training set
                (val_files, "val", False),      # No augmentation for validation
                (test_files, "test", False)     # No augmentation for test set
            ]
            
            for files, split_name, apply_aug in splits_data:
                output_dir = self.output_path / split_name / class_name
                
                for file_path in files:
                    base_name = file_path.stem
                    
                    if apply_aug:
                        # Apply aggressive augmentation for training data
                        augmented_paths = self.aggressive_augmentation(file_path, output_dir, base_name)
                        logger.info(f"Generated {len(augmented_paths)} augmented versions for {file_path.name}")
                    else:
                        # Just copy original for val/test
                        output_path = output_dir / f"{base_name}_original.jpg"
                        try:
                            img = Image.open(file_path)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(output_path, "JPEG", quality=95)
                        except Exception as e:
                            logger.error(f"Error copying {file_path}: {str(e)}")
        
        logger.info("Data preparation completed!")
    
    def generate_dataset_summary(self):
        """Generate a summary of the prepared dataset"""
        summary = {}
        total_images = 0
        
        for split in ["train", "val", "test"]:
            summary[split] = {}
            split_total = 0
            
            for class_name in self.classes:
                class_path = self.output_path / split / class_name
                if class_path.exists():
                    count = len(list(class_path.glob("*.jpg")))
                    summary[split][class_name] = count
                    split_total += count
            
            summary[split]["total"] = split_total
            total_images += split_total
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        
        for split in ["train", "val", "test"]:
            print(f"\n{split.upper()} SET:")
            for class_name in self.classes:
                count = summary[split].get(class_name, 0)
                print(f"  {class_name}: {count} images")
            print(f"  Total: {summary[split]['total']} images")
        
        print(f"\nTOTAL IMAGES: {total_images}")
        print("="*50)
        
        return summary

def main():
    """Main function to run data preparation"""
    # Initialize data preparator
    preparator = DataPreparator()
    
    # Create YOLOv8 directory structure
    preparator.create_yolo_structure()
    
    # Split data and apply augmentation
    preparator.split_and_augment_data(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        augment_factor=3  # Will generate ~3x more training images
    )
    
    # Generate summary
    summary = preparator.generate_dataset_summary()
    
    print("\n✅ Data preparation completed successfully!")
    print("📁 Dataset is ready for YOLOv8 training in the 'yolo_dataset' directory")

if __name__ == "__main__":
    main()
