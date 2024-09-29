import cv2
from picamera2 import Picamera2
import time

image_size = (640, 480)
result_path = "./pics/"

# Initialize stereo cameras using Picamera2
print("Initializing cameras...")
left_cam = Picamera2(0)  # Camera in cam0 port (left camera)
right_cam = Picamera2(1)  # Camera in cam1 port (right camera)

# Configure both cameras with a lower resolution to reduce resource usage
left_cam.configure(left_cam.create_preview_configuration(main={"size": image_size, "format": "RGB888"}))
right_cam.configure(right_cam.create_preview_configuration(main={"size":image_size, "format": "RGB888"}))

# Start the cameras
left_cam.start()
right_cam.start()
print("Cameras started successfully.")

while True:
    # Capture frames from both cameras
    left_frame = left_cam.capture_array()
    right_frame = right_cam.capture_array()

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF

    if key != 255:  # 255 means no key was pressed
        print(f"Key pressed: {chr(key)}")

    if key == ord('p'):
        print("Photo capture initiated...")
        # Save the captured frames
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        left_filename = f"{result_path}/left_{timestamp}.png"
        right_filename = f"{result_path}/right_{timestamp}.png"

        cv2.imwrite(left_filename, left_frame)
        cv2.imwrite(right_filename, right_frame)
        print(f"Images saved as {left_filename} and {right_filename}")

    if key == ord('q'):
        print("Quit key pressed. Exiting...")
        break

# Stop the cameras
left_cam.stop()
right_cam.stop()
print("Cameras stopped. Program exited successfully.")
