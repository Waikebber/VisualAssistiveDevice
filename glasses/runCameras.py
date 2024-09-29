"""
This file runs both cameras connected to the rasberry pi through a python file. 
The primary usage of the file is for testing camera connections.
"""
from picamera2 import Picamera2
import cv2

# Create Picamera2 objects for both cameras
cam0 = Picamera2(0)  # Camera in cam0 port
cam1 = Picamera2(1)  # Camera in cam1 port

# Configure and start both cameras
cam0.configure(cam0.create_preview_configuration())
cam1.configure(cam1.create_preview_configuration())

cam0.start()
cam1.start()

# Function to capture frames from both cameras and display them
while True:
    # Capture frames from both cameras
    frame0 = cam0.capture_array()
    frame1 = cam1.capture_array()

    # Display the frames
    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop cameras and close windows
cam0.stop()
cam1.stop()
cv2.destroyAllWindows()
