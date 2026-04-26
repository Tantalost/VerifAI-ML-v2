import os
from datasets import load_dataset
from tqdm import tqdm

def download_sdxl_subset(dataset_dir="dataset/ai_generated", num_images=5000):
    """
    Downloads a subset of high-resolution AI-generated images (Stable Diffusion XL).
    """
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Directory ready: {dataset_dir}")
    print(f"Connecting to Hugging Face to fetch {num_images} AI images...")

    try:
        # Load a modern, script-free dataset natively sgbupported by Hugging Face
        dataset = load_dataset("ash12321/sdxl-generated-10k", split=f"train[:{num_images}]")
        
        print(f"Successfully connected! Saving images to disk...")
        
        # Iterate through the dataset and save each image
        for i, item in enumerate(tqdm(dataset, desc="Saving AI Images")):
            # The image object is stored in the 'image' key
            image = item['image']
            
            # Convert to RGB to ensure no weird alpha channel (transparency) issues
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Save the image
            filename = os.path.join(dataset_dir, f"ai_sdxl_{i}.jpg")
            image.save(filename, "JPEG", quality=95)
            
        print(f"\nSuccess! {num_images} AI-generated images are now sitting in the '{dataset_dir}' folder.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    download_sdxl_subset()