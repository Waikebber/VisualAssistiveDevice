"""
This file splits the horizontally connected images into left and right images.
Images are stored in the ouput folder like below:
output_folder
 └── input_folder (basename)
        ├── raw
        ├── left
        └── right
"""

import os
import cv2
import shutil

INPUT_FOLDER = "../data/dataset1/RAW/"
OUTPUT_FOLDER =  "../data/stereo_images/scenes/"

def split_and_save_images(input_folder, output_folder):
    """
    Splits images into left and right halves, saves them with appropriate naming conventions,
    and organizes them into subdirectories within the output folder.
    
    Args:
        input_folder (str): The path to the folder containing the images.
        output_folder (str): The path to the output folder where split images will be saved.
    """
    # Default folder name to the name of the input folder
    folder_name = os.path.basename(os.path.normpath(input_folder))
    scene_dir = os.path.join(output_folder, folder_name)
    raw_folder = os.path.join(scene_dir, "raw")
    left_folder = os.path.join(scene_dir, "left")
    right_folder = os.path.join(scene_dir, "right")
    
    # Ensure all necessary subdirectories exist
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    
    count = 1  # Counter for images without "scene" prefix
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            file_path = os.path.join(input_folder, filename)
            
            # Load the stereo image (both left and right views combined)
            pair_img = cv2.imread(file_path, -1)
            if pair_img is None:
                print(f"Failed to load image: {filename}")
                continue
            
            # Compute dimensions for splitting
            actual_height, actual_width = pair_img.shape[:2]
            half_width = actual_width // 2
            
            # Determine naming convention
            if filename.startswith("scene"):
                try:
                    parts = filename.split('_')
                    dimensions = parts[1]
                    image_number = parts[2].split('.')[0]  # Extract image number
                except (IndexError, ValueError):
                    print(f"Failed to parse filename: {filename}")
                    continue
            else:
                # For non-"scene" images, use the count as the image number
                image_number = str(count)
                count += 1
            
            # Save the raw image to the raw folder
            raw_output_path = os.path.join(raw_folder, f"{image_number}.png")
            shutil.copy(file_path, raw_output_path)
            
            # Split the stereo pair into left and right images
            img_left = pair_img[:, :half_width]  # Left image
            img_right = pair_img[:, half_width:]  # Right image
            
            # Save the split images to respective directories with image number
            left_output_path = os.path.join(left_folder, f"{image_number}.png")
            right_output_path = os.path.join(right_folder, f"{image_number}.png")
            
            cv2.imwrite(left_output_path, img_left)
            cv2.imwrite(right_output_path, img_right)
            
            print(f"Processed and saved: {filename} as {image_number}")

    print("Processing complete.")