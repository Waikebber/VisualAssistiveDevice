import cv2
from picamera2 import Picamera2
import time
import os

img_size = (640, 480)
save_path = "/pics/"

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Directory {save_path} created.")

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
                left_filename = os.path.join(save_path, f"left_{timestamp}.png")
                right_filename = os.path.join(save_path, f"right_{timestamp}.png")

                # Save the images
                cv2.imwrite(left_filename, left_frame)
                cv2.imwrite(right_filename, right_frame)

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
