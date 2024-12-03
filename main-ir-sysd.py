import os, json, multiprocessing
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from speakers import speak
from stereo_calibration.camera.cam_config import initialize_camera
from image_rec.img_rec import ImgRec
from stereo_calibration.rectify import get_closest_distance, make_colored_distance_map, rectify_imgs, make_disparity_map
from image_rec.stereoImgRec import create_detection_image, calculate_object_distances
from multiprocessing import Process, Queue
from queue import Empty
from dataclasses import dataclass
from typing import Any
from enum import Enum
import logging
import signal
import sys
import threading
from multiprocessing.synchronize import Event

CONFIDENCE = 0.6
THRESHOLD = 3.5   # Threshold in meters (2.5m)
CONFIG_FILE = "stereo_calibration/cam_config.json"
CALIB_RESULTS = 'data/stereo_images/scenes/calibration_results'
SAVE_OUTPUT = True
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'output.png'
DISPLAY_RATIO = 1  # Scaling factor for display
BORDER = 50        # Border to ignore for depth map calculations

################# SystemD ################# 

# Function to handle termination signals
def handle_exit(signum, frame):
    logging.info("Received termination signal. Exiting gracefully...")
    sys.exit(0)

# Register signal handlers for SIGTERM and SIGINT
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
################# Import Configurations ################# 

# Load configuration from config.json
config_path = CONFIG_FILE
if not os.path.isfile(config_path):
    logging.warning(f"Configuration file {config_path} not found.")
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
BASELINE = int(config['baseline_length_mm']) / 1000  # Baseline in m (60mm)
FOCAL = int(config['focal_length_mm']) / 1000        # Focal length in m (2.6mm)
SENSOR_WIDTH = float(config['cmos_size_m'])          # Sensor width in m (1/4in)
H_FOV = int(config['field_of_view']['horizontal'])   # Horizontal field of view (73deg)
scale_ratio = float(config['scale_ratio'])           # Image scaling ratio (0.5)
cam_width = int(config['image_width'])
cam_height = int(config['image_height'])

DISTANCE_SCALE = BASELINE * 5 / 8

# Camera resolution / image scaling / focal length
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
logging.info(f"Used camera resolution: {str(cam_width)} x {str(cam_height)}")
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
logging.info(f"Image resolution: {str(img_width)} x {str(img_height)}")
print("Image resolution: " + str(img_width) + " x " + str(img_height))

# Initialize the image recognition model and distance calculator
img_recognizer = ImgRec()

# Shutsdown the System
shutdown_event = multiprocessing.Event()

################# Audio Processing #################
class Priority(Enum):
    HIGH = 1    # For button press messages
    LOW = 2     # For regular distance updates

@dataclass
class PriorityMessage:
    priority: Priority
    text: str

# Create two separate queues for different priority levels
high_priority_queue = Queue()
low_priority_queue = Queue()

def audio_worker():
    """Worker process that handles messages based on priority"""
    while not shutdown_event.is_set():
        try:
            # Always check high priority queue first
            try:
                # Non-blocking check of high priority queue
                message = high_priority_queue.get_nowait()
                speak(message.text, 3, 70)
                continue  # Go back to start of loop to check for more high priority messages
            except Empty:
                pass

            # Only check low priority queue if no high priority messages
            message = low_priority_queue.get(timeout=1)
            speak(message.text, 3, 70)
            
        except Empty:
            continue
        except Exception as e:
            if not shutdown_event.is_set():
                logging.warning(f"Error in audio worker: {e}")
                print(f"Error in audio worker: {e}")

audio_process = None
def speak_async(text, priority=Priority.LOW):
    """Add text to the appropriate priority queue"""
    try:
        message = PriorityMessage(priority=priority, text=text)
        if priority == Priority.HIGH:
            high_priority_queue.put(message)
        else:
            low_priority_queue.put(message)
    except Exception as e:
        logging.warning(f"Error queueing audio: {e}")
        print(f"Error queueing audio: {e}")

audio_worker_process = Process(target=audio_worker, daemon=True)
audio_worker_process.start()
################# GPIO Setup #################
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 26
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button_press_action():
    global current_distances, rectified_pair
    distances_on_button_press = current_distances
    current_pair = rectified_pair
    left_rectified = current_pair[0]
    
    if current_pair is None or distances_on_button_press is None:
        print("No frames available yet")
        return
        
    print("Button pressed - performing object detection and distance measurement")
    
    detected_objects = img_recognizer.predict_frame(left_rectified)
    print('OBJECTS')
    print(detected_objects)

    if SAVE_OUTPUT:
        output = create_detection_image(distances_on_button_press, detected_objects, BORDER)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FILE), output)
    
    # Calculate distances for detected objects using current distance map
    object_distances = calculate_object_distances(distances_on_button_press, detected_objects, border=0, percentile=20, confidence_threshold=CONFIDENCE)
    
    # Process and announce detected objects within threshold
    for obj_name, distance_val in object_distances:
        if distance_val < THRESHOLD:
            message = f"Detected {obj_name} at {distance_val:.2f} meters"
            print(message)
            speak_async(message, Priority.HIGH)
        else:
            message = f"{obj_name} further than {THRESHOLD} meters"
            print(message)
            speak_async(message, Priority.HIGH)

def on_button_press(channel):
    p = multiprocessing.Process(target=button_press_action)
    p.start()

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=on_button_press, bouncetime=200)

################# Camera Processing #################

# Initialize the cameras
camera_left = initialize_camera(1, img_width, img_height)
camera_right = initialize_camera(0, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

# Initialize interface windows
"""
cv2.namedWindow("Depth Map")
cv2.moveWindow("Depth Map", 50, 100)
cv2.namedWindow("Left Camera")
cv2.moveWindow("Left Camera", 450, 100)
cv2.namedWindow("Right Camera")
cv2.moveWindow("Right Camera", 850, 100)
"""

try:
    # Start the main processing loop
    while True:
        # Capture frames
        current_frame_left = camera_left.capture_array()
        current_frame_right = camera_right.capture_array()
    
        # Rectify the stereo pair
        left_rectified, right_rectified, Q, focal_length = rectify_imgs(current_frame_left, current_frame_right, CALIB_RESULTS)

        # Generate the disparity map
        min_disp = 0
        num_disp = 16 * 4  # must be divisible by 16
        block_size = 10
        disparity = make_disparity_map(left_rectified, right_rectified, min_disp, num_disp, block_size)
        rectified_pair = (left_rectified, right_rectified)
        
        # Reduce Noise In Disparity Map
        if disparity is not None:
            kernel = np.ones((3,3), np.uint8)
            disparity = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
            disparity = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
            
        
        # Convert disparity to depth map
        depth_map = cv2.reprojectImageTo3D(disparity, Q)
        current_distances = depth_map[:, :, 2] * DISTANCE_SCALE
        
        # In your main loop:
        closest_distance, closest_coordinates = get_closest_distance(
            distances=current_distances,
            disparity_map=disparity,  # Pass the raw disparity map
            min_thresh=1,
            max_thresh=5,
            border=BORDER,
            topBorder=100,
            min_region_area=6000  # Adjust based on your typical object size
        )
        
        if closest_distance is not None and closest_coordinates is not None and closest_distance < THRESHOLD:
            message = f'Closest distance: {closest_distance:.2f} meters'
            logging.info(f'{message} at {closest_coordinates}')
            print(f'{message} at {closest_coordinates}')
            speak_async(message, Priority.LOW)
        
        # Create a colored depth map for visualization
        depth_map_colored = make_colored_distance_map(current_distances, min_distance=0, max_distance=THRESHOLD)
        
        # Draw the closest point on the left rectified image and disparity map
        if closest_coordinates is not None:
            cv2.circle(left_rectified, closest_coordinates, 5, (0, 255, 0), 2)
            cv2.circle(current_distances, closest_coordinates, 5, (0, 255, 0), 2)

        # Display frames
        """
        cv2.imshow("Left Camera", left_rectified)
        cv2.imshow("Right Camera", right_rectified)
        cv2.imshow("Depth Map", depth_map_colored)
        """


        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except Exception as e:
    # Log any errors that occur during the loop
    print(f"Error during processing: {e}")
    logging.warning(f"Error during processing: {e}")

finally:
    # Ensure cleanup happens no matter what
    shutdown_event.set()
    audio_worker_process.join(timeout=2)
    if audio_worker_process.is_alive():
        audio_worker_process.terminate()
    camera_left.stop()
    camera_right.stop()
    # cv2.destroyAllWindows()
