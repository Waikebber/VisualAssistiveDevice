import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO('yolov8n.pt')  # YOLO nano version for lower resource usage

# Initialize stereo cameras using Picamera2
left_cam = Picamera2(0)  # Camera in cam0 port (left camera)
right_cam = Picamera2(1)  # Camera in cam1 port (right camera)

# Configure both cameras
left_cam.configure(left_cam.create_preview_configuration(main={"format": "RGB888"}))
right_cam.configure(right_cam.create_preview_configuration(main={"format": "RGB888"}))

# Start the cameras
left_cam.start()
right_cam.start()

# Function to display detection results on the frame
def process_frame(frame):
    # Convert the frame from RGB to BGR for OpenCV/YOLO compatibility
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Run the YOLO detection model on the frame
    results = model(frame_bgr)
    
    # Plot the results (bounding boxes, etc.)
    return results[0].plot()

# Initialize time tracking variables
last_detection_time = time.time()
detection_interval = 3  # Set the interval to 3 seconds

while True:
    # Capture frames from both cameras
    left_frame = left_cam.capture_array()
    right_frame = right_cam.capture_array()

    # Get the current time
    current_time = time.time()

    if left_frame is not None and right_frame is not None:
        # Combine the frames side by side for display (without YOLO detection)
        combined = cv2.hconcat([left_frame, right_frame])
        
        # Display the combined frame without detections
        cv2.imshow('Stereo Camera', combined)

        # Perform object detection every 3 seconds
        if current_time - last_detection_time >= detection_interval:
            # Process left and right frames with YOLO model
            left_with_detections = process_frame(left_frame)
            right_with_detections = process_frame(right_frame)

            # Combine the frames side by side with YOLO detections
            combined_with_detections = cv2.hconcat([left_with_detections, right_with_detections])
            
            # Display the combined frame with YOLO detections
            cv2.imshow('Stereo Camera with YOLO', combined_with_detections)

            # Update the last detection time
            last_detection_time = current_time

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the cameras and close windows
left_cam.stop()
right_cam.stop()
cv2.destroyAllWindows()
