import os
import requests
import zipfile
from tqdm import tqdm

def download_coco_subset(dataset_dir="dataset/real", num_images=5000):
    """
    Downloads and extracts the COCO 2017 Validation dataset (5,000 real images).
    """
    # Official COCO dataset URL for the 2017 validation images
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = "val2017.zip"
    
    # Create the target directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Directory ready: {dataset_dir}")

    # 1. Download the ZIP file with a progress bar
    if not os.path.exists(zip_path):
        print(f"Downloading 5,000 real images from COCO...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc=zip_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        print("Zip file already exists. Skipping download.")

    # 2. Extract the ZIP file
    print("Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract everything into our target directory
        zip_ref.extractall("dataset/temp_coco")
        
    # Move files from the extracted 'val2017' folder to our 'dataset/real' folder
    extracted_folder = "dataset/temp_coco/val2017"
    for filename in tqdm(os.listdir(extracted_folder), desc="Moving files"):
        os.rename(
            os.path.join(extracted_folder, filename),
            os.path.join(dataset_dir, filename)
        )
        
    # Clean up the leftover zip and temp folders
    os.rmdir(extracted_folder)
    os.rmdir("dataset/temp_coco")
    os.remove(zip_path)
    
    print(f"\nSuccess! 5,000 real images are now sitting in the '{dataset_dir}' folder.")

if __name__ == "__main__":
    download_coco_subset()