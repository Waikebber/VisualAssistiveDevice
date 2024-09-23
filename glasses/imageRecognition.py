"""
This file is used to see the real-time image detection ability of the cameras.
If the rasberry pi is directly connected to a monitor via hdmi, you should be able to see real-time detection.
"""
import cv2
from ultralytics import YOLO
import psutil

model = YOLO('yolov8n.pt')  # YOLO nano version for lower resource usage

# Initialize stereo cameras
left_cam = cv2.VideoCapture(0)   # Left camera
right_cam = cv2.VideoCapture(1)  # Right camera

# Function to display detection results on the frame
def process_frame(frame):
    results = model(frame)  
    return results[0].plot() 

while True:
    retL, left_frame = left_cam.read()
    retR, right_frame = right_cam.read()

    if retL and retR:
        # Process left and right images
        left_with_detections = process_frame(left_frame)
        right_with_detections = process_frame(right_frame)

        # Combine the frames side by side
        combined = cv2.hconcat([left_with_detections, right_with_detections])
        cv2.imshow('Stereo Camera with YOLO', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
