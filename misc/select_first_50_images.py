'''Script to select and copy the first 50 images from test_images folder'''

import os
import shutil
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import TEST_IMAGES_PATH, FIRST_50_IMAGES_PATH
IMAGE_PREFIX = "image_"
IMAGE_EXTENSION = ".png"

# Configuration
source_path = TEST_IMAGES_PATH
destination_path = FIRST_50_IMAGES_PATH

if __name__ == "__main__":
    # Create destination folder if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    # Copy first 50 images (image_0.png to image_49.png)
    copied_count = 0
    for i in range(50):
        image_name = f"{IMAGE_PREFIX}{i}{IMAGE_EXTENSION}"
        source_file = source_path / image_name
        destination_file = destination_path / image_name

        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
            copied_count += 1
            print(f"Copied: {image_name}")
        else:
            print(f"Warning: {image_name} not found in source directory")

    print(f"\nTotal images copied: {copied_count}/50")
    print(f"Destination: {destination_path}")


