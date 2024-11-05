import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time
from ImgRec import ImgRec

# Reduce OpenCV's thread usage to avoid overload
cv2.setNumThreads(1)

# Initialize ImgRec object
print("Initializing ImgRec for YOLO model...")
img_rec = ImgRec()
print("ImgRec initialized successfully.")

# Initialize stereo cameras using Picamera2
print("Initializing cameras...")
left_cam = Picamera2(0)  # Camera in cam0 port (left camera)
right_cam = Picamera2(1)  # Camera in cam1 port (right camera)

# Configure both cameras with a lower resolution to reduce resource usage
left_cam.configure(left_cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
right_cam.configure(right_cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))

# Start the cameras
left_cam.start()
right_cam.start()
print("Cameras started successfully.")

# Function to process and save detection results for a frame using ImgRec
def process_frame_and_save(frame, filename, confidence_threshold=0.5):
    print(f"Processing frame and saving to {filename}...")
    # Run the ImgRec detection model on the frame and interpret results
    detections = img_rec.predict_frame(frame, confidence_threshold=confidence_threshold)
    
    # Convert the frame from RGB to BGR for OpenCV compatibility (if needed)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Annotate the frame with detections
    for obj_name, bounding_box, confidence in detections:
        x_min, y_min, x_max, y_max = map(int, bounding_box)
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{obj_name} ({confidence:.2f})"
        cv2.putText(frame_bgr, label, (x_min, y_min - 10 if y_min > 20 else y_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the annotated frame to a file
    cv2.imwrite(filename, frame_bgr)
    print(f"Classified image saved as {filename}")

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
        # Take a photo and classify it, then save the result
        if left_frame is not None and right_frame is not None:
            # Save classified frames with YOLO detections as PNG files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            left_filename = f"left_classified_{timestamp}.png"
            right_filename = f"right_classified_{timestamp}.png"
            
            # Process and save each frame
            process_frame_and_save(left_frame, left_filename)
            process_frame_and_save(right_frame, right_filename)

    if key == ord('q'):
        print("Quit key pressed. Exiting...")
        break

# Stop the cameras
left_cam.stop()
right_cam.stop()
print("Cameras stopped. Program exited successfully.")
