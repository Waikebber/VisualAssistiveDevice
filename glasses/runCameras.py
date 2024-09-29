import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # YOLO nano version for lower resource usage
print("YOLO model loaded successfully.")

# Initialize stereo cameras using Picamera2
print("Initializing cameras...")
left_cam = Picamera2(0)  # Camera in cam0 port (left camera)
right_cam = Picamera2(1)  # Camera in cam1 port (right camera)

# Configure both cameras
left_cam.configure(left_cam.create_preview_configuration(main={"format": "RGB888"}))
right_cam.configure(right_cam.create_preview_configuration(main={"format": "RGB888"}))

# Start the cameras
left_cam.start()
right_cam.start()
print("Cameras started successfully.")

# Function to save detection results on the frame
def process_frame_and_save(frame, filename):
    print(f"Processing frame and saving to {filename}...")
    # Convert the frame from RGB to BGR for OpenCV/YOLO compatibility
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Run the YOLO detection model on the frame
    results = model(frame_bgr)
    
    # Plot the results (bounding boxes, etc.)
    result_img = results[0].plot()
    
    # Save the result to a file
    cv2.imwrite(filename, result_img)
    print(f"Classified image saved as {filename}")

while True:
    # Capture frames from both cameras
    left_frame = left_cam.capture_array()
    right_frame = right_cam.capture_array()

    if left_frame is not None and right_frame is not None:
        # Combine the frames side by side for display
        combined = cv2.hconcat([left_frame, right_frame])
        
        # For debugging, print that frames are being captured
        print("Displaying real-time video. Press 'p' to take a picture or 'q' to quit.")
        
        # Display the combined frame in real-time
        cv2.imshow('Stereo Camera', combined)

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF

    # Debugging print statements for keypresses
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

# Stop the cameras and close windows
left_cam.stop()
right_cam.stop()
cv2.destroyAllWindows()
print("Cameras stopped and windows closed. Program exited successfully.")
