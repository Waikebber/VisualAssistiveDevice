"""
    This file runs both cameras and takes images on button press (p).
    THIS FILE WILL PROMPT FOR 15 IMAGES FROM EACH CAMERA. 30 TOTAL IMAGES.
    The program exits on button press (q).
"""

from picamera2 import Picamera2
import time
import os
from PIL import Image
import cv2

# Define the base folder where images will be stored
ROOT = "images"
IMGS_NEEDED = 15
IMG_SIZE = (640, 480)

# Initialize stereo cameras using Picamera2
print("Initializing cameras...")
left_cam = Picamera2(0)  # Camera in cam1 port (left camera)
right_cam = Picamera2(1)  # Camera in cam0 port (right camera)

# Configure both cameras with a lower resolution to reduce resource usage
left_cam.configure(left_cam.create_preview_configuration(main={"size": IMG_SIZE, "format": "RGB888"}))
right_cam.configure(right_cam.create_preview_configuration(main={"size": IMG_SIZE, "format": "RGB888"}))

# Set white balance
left_cam.set_controls({"AwbEnable":True})
right_cam.set_controls({"AwbEnable":True})

# Start the cameras
left_cam.start()
right_cam.start()
print("Cameras started successfully.")

# Convert numpy array to image using Pillow
def save_image(image_array, filename):
    image = Image.fromarray(image_array)  # Convert from numpy array to PIL Image
    image.save(filename)  # Save the image using Pillow
print("Press 'p' to take a photo or 'q' to quit: ")
# Main loop
try:
    count = 0
    while count < IMGS_NEEDED:
        # Capture frames from both cameras
        left_frame = left_cam.capture_array()
        right_frame = right_cam.capture_array()

        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)
                
        # Prompt for keypress
        key = cv2.waitKey(1) & 0xFF

        # Set and Create the Save Directories
        save_folder_left = os.path.join(ROOT, "stereoLeft")
        save_folder_right = os.path.join(ROOT, "stereoRight")
        if not os.path.exists(save_folder_left):
            os.makedirs(save_folder_left)
            print(f"Directory {save_folder_left} created.")
        if not os.path.exists(save_folder_right):
            os.makedirs(save_folder_right)
            print(f"Directory {save_folder_right} created.")
            
        
        if key == ord('p'):
            print("Photo capture initiated...")
            if left_frame is not None and right_frame is not None:
                # Define filenames for saving images
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                left_filename = os.path.join(save_folder_left, f"{timestamp}.png")
                right_filename = os.path.join(save_folder_right, f"{timestamp}.png")

                # Save images using Pillow
                save_image(left_frame, left_filename)
                save_image(right_frame, right_filename)

                # Print the full absolute path of the saved images
                print(f"Images saved at:\nLeft: {os.path.abspath(left_filename)}\nRight: {os.path.abspath(right_filename)}")
                
                count = count + 1
                print(f"Taken {count} of {IMGS_NEEDED}")
                print("Press 'p' to take a photo or 'q' to quit: ")
        elif key == ord('q'):
            print("Quit key pressed. Exiting...")
            break

finally:
    # Stop the cameras
    left_cam.stop()
    right_cam.stop()
    print("Cameras stopped. Program exited successfully.")
