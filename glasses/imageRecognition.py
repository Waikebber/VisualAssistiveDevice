"""
This file is used to see the real-time image detection ability of the cameras.
If the Raspberry Pi is directly connected to a monitor via HDMI, you should be able to see real-time detection.
"""
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # YOLO nano version for lower resource usage

# Initialize stereo cameras using Picamera2
left_cam = Picamera2(0)  # Camera in cam0 port (left camera)
right_cam = Picamera2(1)  # Camera in cam1 port (right camera)

# Configure both cameras
left_cam.configure(left_cam.create_preview_configuration())
right_cam.configure(right_cam.create_preview_configuration())

# Start the cameras
left_cam.start()
right_cam.start()

# Function to display detection results on the frame
def process_frame(frame):
    results = model(frame)  
    return results[0].plot() 

while True:
    # Capture frames from both cameras
    left_frame = left_cam.capture_array()
    right_frame = right_cam.capture_array()

    if left_frame is not None and right_frame is not None:
        # Process left and right images with YOLO model
        left_with_detections = process_frame(left_frame)
        right_with_detections = process_frame(right_frame)

        # Combine the frames side by side
        combined = cv2.hconcat([left_with_detections, right_with_detections])
        cv2.imshow('Stereo Camera with YOLO', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the cameras and close windows
left_cam.stop()
right_cam.stop()
cv2.destroyAllWindows()
