from picamera2 import Picamera2
import time
import os
from PIL import Image
import numpy as np

img_size = (640, 480)

# Initialize stereo cameras using Picamera2
print("Initializing cameras...")
left_cam = Picamera2(0)  # Camera in cam0 port (left camera)
right_cam = Picamera2(1)  # Camera in cam1 port (right camera)

# Configure both cameras with a lower resolution to reduce resource usage
left_cam.configure(left_cam.create_preview_configuration(main={"size": img_size, "format": "RGB888"}))
right_cam.configure(right_cam.create_preview_configuration(main={"size": img_size, "format": "RGB888"}))

# Start the cameras
left_cam.start()
right_cam.start()
print("Cameras started successfully.")

# Convert numpy array to image using PIL
def save_image(image_array, filename):
    image = Image.fromarray(image_array)  # Convert from numpy array to PIL Image
    image.save(filename)  # Save the image using Pillow

# Main loop
try:
    while True:
        # Prompt for keypress
        key = input("Press 'p' to take a photo or 'q' to quit: ")

        if key == 'p':
            print("Photo capture initiated...")

            # Capture frames from both cameras
            left_frame = left_cam.capture_array()
            right_frame = right_cam.capture_array()

            if left_frame is not None and right_frame is not None:
                # Save the captured frames
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                left_filename = f"left_{timestamp}.png"
                right_filename = f"right_{timestamp}.png"

                # Save images using Pillow
                save_image(left_frame, left_filename)
                save_image(right_frame, right_filename)

                # Print the full absolute path of the saved images
                print(f"Images saved at:\nLeft: {os.path.abspath(left_filename)}\nRight: {os.path.abspath(right_filename)}")

        elif key == 'q':
            print("Quit key pressed. Exiting...")
            break

finally:
    # Stop the cameras
    left_cam.stop()
    right_cam.stop()
    print("Cameras stopped. Program exited successfully.")
