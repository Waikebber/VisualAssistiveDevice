import keyboard  # To handle keypresses
import cv2
from picamera2 import Picamera2
import time

img_size =  (640, 480)
save_path = "./pics/"

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
        # Check for 'p' key press to capture images
        if keyboard.is_pressed('p'):
            print("Photo capture initiated...")

            # Capture frames from both cameras
            left_frame = left_cam.capture_array()
            right_frame = right_cam.capture_array()

            if left_frame is not None and right_frame is not None:
                # Save the captured frames
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                left_filename = f"{save_path}left_{timestamp}.png"
                right_filename = f"{save_path}right_{timestamp}.png"

                cv2.imwrite(left_filename, left_frame)
                cv2.imwrite(right_filename, right_frame)
                print(f"Images saved as {left_filename} and {right_filename}")

            # To avoid multiple captures on a single press, wait for release
            keyboard.wait('p', suppress=True)

        # Check for 'q' key press to quit
        if keyboard.is_pressed('q'):
            print("Quit key pressed. Exiting...")
            break

finally:
    # Stop the cameras
    left_cam.stop()
    right_cam.stop()
    print("Cameras stopped. Program exited successfully.")
