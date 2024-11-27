import time, os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime
from math import tan, pi
from speakers import speak
from stereo_calibration.camera.cam_config import initialize_camera
import multiprocessing
import RPi.GPIO as GPIO
from image_rec.img_rec import ImgRec
from distance_calculator.DistanceCalculator import DistanceCalculator
from stereo_calibration.tuning_helper import load_map_settings_with_sgbm, load_map_settings_with_sbm
import logging

USE_SGBM = True
CONFIDENCE = 0.6
THRESHOLD = 2.5   # Threshold in meters (2.5m)
CONFIG_FILE = "stereo_calibration/cam_config.json"
SETTINGS_FILE = "stereo_calibration/3dmap_set.txt"
CALIB_RESULTS = 'stereo_calibration/calib_result'
SAVE_OUTPUT = True
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'output.png'
DISPLAY_RATIO = 0.5  # Scaling factor for display


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

# Load configuration from config.json
config_path = CONFIG_FILE
if not os.path.isfile(config_path):
    logging.warning(f"Configuration file {config_path} not found.")
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

BASELINE = 62/1000 #int(config['baseline_length_mm']) / 1000  # Baseline in m (60mm)
FOCAL = int(config['focal_length_mm']) / 1000        # Focal length in m (2.6mm)
SENSOR_WIDTH = float(config['cmos_size_m'])          # Sensor width in m (1/4in)
H_FOV = int(config['field_of_view']['horizontal'])   # Horizontal field of view (73deg)
scale_ratio = float(config['scale_ratio'])           # Image scaling ratio (0.5)
cam_width = int(config['image_width'])
cam_height = int(config['image_height'])

# Camera resolution / image scaling / focal length
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))
logging.info(f"Used camera resolution: {str(cam_width)} x {str(cam_height)}")
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Image resolution: " + str(img_width) + " x " + str(img_height))
logging.info(f"Image resolution: {str(img_width)} x {str(img_height)}")
focal_length_px = (img_width * 0.5) / tan(H_FOV * 0.5 * pi / 180) # Focal length in px with FOV
print("Focal length: " + str(focal_length_px) + " px")
logging.info(f"Focal length: {str(focal_length_px)} px")

# Initialize the image recognition model and distance calculator
img_recognizer = ImgRec()
distance = DistanceCalculator(BASELINE, focal_length_px)

# GPIO Setup
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 21
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button_press_action():
    global current_disparity, rectified_pair, distance
    disparity_on_button_press = current_disparity
    current_pair = rectified_pair
    
    if current_pair is None or disparity_on_button_press is None:
        print("No frames available yet")
        logging.info("No frames available yet")
        return
        
    print("Button pressed - performing object detection and distance measurement")
    logging.info("Button pressed - performing object detection and distance measurement")
    
    # Convert rectified grayscale to BGR for object detection
    rectified_color = cv2.cvtColor(current_pair[0], cv2.COLOR_GRAY2BGR)
    
    detected_objects = img_recognizer.predict_frame(rectified_color)
    print('OBJECTS')
    print(detected_objects)
    logging.info(f"OBJECTS: {detected_objects}")
    
    if SAVE_OUTPUT:
        output = distance.create_detection_image(disparity_on_button_press, detected_objects)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FILE), output)
    
    # Calculate distances for detected objects using current disparity
    object_distances = distance.calculate_object_distances(disparity_on_button_press, detected_objects)
    
    # Process and announce detected objects within threshold
    for obj_name, distance_val, confidence, coords in object_distances:
        if distance_val < THRESHOLD and CONFIDENCE <= confidence:
            distance_ft = round(distance_val * 3.281, 2)
            message = f"Warning: {obj_name} detected at {distance_val:.2f} meters ({distance_ft} feet)"
            print(message)
            logging.info(message)
            speak_async(message)

def on_button_press(channel):
    p = multiprocessing.Process(target=button_press_action)
    p.start()

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=on_button_press, bouncetime=200)

# Initialize the cameras
camera_left = initialize_camera(1, img_width, img_height)
camera_right = initialize_camera(0, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
logging.info("Read calibration data and rectifying stereo pair...")
calibration = StereoCalibration(input_folder=CALIB_RESULTS)

# Initialize interface windows
cv2.namedWindow("Depth Map")
cv2.moveWindow("Depth Map", 50, 100)
cv2.namedWindow("Disparity Map")
cv2.moveWindow("Disparity Map", 200, 100)
cv2.namedWindow("Left Camera")
cv2.moveWindow("Left Camera", 450, 100)
cv2.namedWindow("Right Camera")
cv2.moveWindow("Right Camera", 850, 100)

# Load map settings and initialize the StereoBM (Block Matching) object with updated parameters
if USE_SGBM:
    sbm = load_map_settings_with_sgbm(SETTINGS_FILE)
else:
    sbm = load_map_settings_with_sbm(SETTINGS_FILE)

audio_process = None
def speak_async(text):
    global audio_process
    if audio_process is None or not audio_process.is_alive():
        audio_process = multiprocessing.Process(target=speak, args=(text, 3, 90))
        audio_process.start()

# def stereo_depth_map(rectified_pair):
    # dmLeft = rectified_pair[0]
    # dmRight = rectified_pair[1]

    # # Compute the disparity map
    # disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
    # lower_bound = np.percentile(disparity, 5)
    # upper_bound = np.percentile(disparity, 95)
    # local_min = disparity.min()
    # local_max = disparity.max()

    # # Clip and reduce noise in disparity map
    # disparity = np.clip(disparity, lower_bound, upper_bound)
    # disparity = distance.reduce_noise(disparity)

    # # Improved normalization and visualization of the disparity map
    # disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    # disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    # disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

    # # Calculate distance to the center pixel
    # center_pixel_distance = distance.solve_center_pixel_distance(disparity)

    # # Annotate the center pixel distance on the disparity map
    # h, w = disparity.shape
    # center_y, center_x = h // 2, w // 2
    # cv2.putText(
        # disparity_color,
        # f"Center: {center_pixel_distance:.2f} units",
        # (10, 30),
        # cv2.FONT_HERSHEY_SIMPLEX,
        # 1.0,
        # (0, 255, 0),
        # 2,
    # )
    # cv2.circle(disparity_color, (center_x, center_y), 5, (0, 0, 255), -1)

    # # Display the colored disparity map
    # cv2.imshow("Disparity Map (Color)", disparity_color)

    # return disparity_color, disparity

# Add mouse callback for disparity map
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity_normalized = param
        if disparity_normalized is not None:
            disparity_value = disparity_normalized[y][x]
            
            # Ensure disparity is non-zero to avoid division by zero
            if disparity_value > 0:
                distance = (BASELINE * focal_length_px) / disparity_value
                print(f"Distance at ({x}, {y}): {distance:.2f} meters  Disparity {disparity_value}")
                logging.info(f"Distance at ({x}, {y}): {distance:.2f} meters  Disparity {disparity_value}")
            else:
                print(f"Disparity at ({x}, {y}) is zero or invalid, cannot calculate distance.")
                logging.warning(f"Disparity at ({x}, {y}) is zero or invalid, cannot calculate distance.")


# Modify stereo_depth_map function to return normalized disparity
def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]

    # Compute the disparity map
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
    lower_bound = np.percentile(disparity, 5)
    upper_bound = np.percentile(disparity, 95)
    local_min = disparity.min()
    local_max = disparity.max()

    # Clip and reduce noise in disparity map
    disparity = np.clip(disparity, lower_bound, upper_bound)
    disparity = distance.reduce_noise(disparity)

    # Improved normalization and visualization of the disparity map
    disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

    return disparity_color, disparity

# Update main loop to include mouse callback
try:
    # Start the main processing loop
    while True:
        # Capture frames
        current_frame_left = camera_left.capture_array()
        current_frame_right = camera_right.capture_array()
    
        # Convert to grayscale
        imgLeft = cv2.cvtColor(current_frame_left, cv2.COLOR_BGR2GRAY)
        imgRight = cv2.cvtColor(current_frame_right, cv2.COLOR_BGR2GRAY)
    
        # Rectify the stereo pair
        rectified_pair = calibration.rectify((imgLeft, imgRight))
    
        # Generate and display the depth map
        disparity_color, current_disparity = stereo_depth_map(rectified_pair)

        # Set up mouse callback for disparity map
        cv2.imshow("Disparity Map (Color)", disparity_color)
        cv2.setMouseCallback("Disparity Map (Color)", onMouse, current_disparity)

        # Resize and display rectified images
        imgLeft_display = cv2.resize(rectified_pair[0], (0, 0), fx=DISPLAY_RATIO, fy=DISPLAY_RATIO)
        imgRight_display = cv2.resize(rectified_pair[1], (0, 0), fx=DISPLAY_RATIO, fy=DISPLAY_RATIO)
        cv2.imshow("Left Camera", imgLeft_display)
        cv2.imshow("Right Camera", imgRight_display)

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
    camera_left.stop()
    camera_right.stop()
    cv2.destroyAllWindows()
