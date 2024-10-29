"""
This file runs image recognition on 2 (left and right) images in the same directory that are the most recent.
Directory structure should be as follows:
/ basepath (pics) / *time-stamp* / left_*.png
/ basepath (pics) / *time-stamp* / right_*.png
"""
import os
import glob
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model (e.g., YOLOv8n)
model = YOLO('yolov8n.pt')

# Function to get the most recent folder in the pics directory
def get_most_recent_folder(base_folder="pics"):
    # Find all folders in the base folder
    folders = glob.glob(os.path.join(base_folder, "*"))
    
    # Return the most recent folder based on the timestamp in the folder name
    if folders:
        return max(folders, key=os.path.getmtime)
    return None

# Function to run image recognition on both images in the folder
def run_image_recognition(folder):
    # Find the left and right images (assuming they are the only two images in the folder)
    left_image_path = os.path.join(folder, "left_*.png")
    right_image_path = os.path.join(folder, "right_*.png")
    
    # Load the images using PIL
    left_image = Image.open(glob.glob(left_image_path)[0])
    right_image = Image.open(glob.glob(right_image_path)[0])

    # Run YOLO on both images
    results_left = model(left_image)
    results_right = model(right_image)

    # Print detected objects and their counts for each image
    for i, results in enumerate([results_left, results_right]):
        print(f"Image {i + 1}:")
        detected_objects = results[0].boxes.cls  # Get the class IDs
        object_names = [model.names[int(cls)] for cls in detected_objects]  # Convert class IDs to names
        object_count = {name: object_names.count(name) for name in set(object_names)}  # Count occurrences
        print(f"Detected objects and their counts: {object_count}")

    # Display both images with bounding boxes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(results_left[0].plot())
    axs[0].set_title('Left Image with Bounding Boxes')
    axs[0].axis('off')

    axs[1].imshow(results_right[0].plot())
    axs[1].set_title('Right Image with Bounding Boxes')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # Get the most recent folder
    most_recent_folder = get_most_recent_folder()

    if most_recent_folder:
        print(f"Most recent folder: {most_recent_folder}")
        # Run image recognition on the two images in the folder
        run_image_recognition(most_recent_folder)
    else:
        print("No folders found in the base directory.")
