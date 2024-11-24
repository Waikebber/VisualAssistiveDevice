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

# Constants and Configuration
CONFIDENCE = 0.6
THRESHOLD = 2.5   # Threshold in meters (2.5m)
CONFIG_FILE = "stereo_calibration/cam_config.json"
SETTINGS_FILE = "stereo_calibration/3dmap_set.txt"
CALIB_RESULTS = 'stereo_calibration/calib_result'

SAVE_OUTPUT = True
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'output.png'

# Load configuration from config.json
config_path = CONFIG_FILE
if not os.path.isfile(config_path):
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

# Camera resolution / image scaling / focal length
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Image resolution: " + str(img_width) + " x " + str(img_height))
focal_length_px = (img_width * 0.5) / tan(H_FOV * 0.5 * pi / 180) # Focal length in px with FOV
print("Focal length: " + str(focal_length_px) + " px")

# Initialize components
img_recognizer = ImgRec()
distance = DistanceCalculator(BASELINE, focal_length_px)

# GPIO Setup
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 21
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Global variables for frame storage
current_frame_left = None
current_frame_right = None
current_disparity = None
rectified_pair = None

def button_press_action():
    global current_frame_left, current_frame_right, current_disparity, calibration, rectified_pair
    if current_frame_left is None or current_frame_right is None or rectified_pair is None:
        print("No frames available yet")
        return

    print("Button pressed - performing object detection and distance measurement")
    
    # Use the already rectified left image for detection
    rectified_left = cv2.cvtColor(rectified_pair[0], cv2.COLOR_GRAY2BGR)
    
    # Perform object detection on the rectified left image
    detected_objects = img_recognizer.predict_frame(rectified_left)
    
    # Process detected objects
    bounding_boxes = [
        (x, y, w, h, label, distance.calculate_distance(x + w//2, y + h//2, current_disparity))
        for label, (x, y, w, h), confidence in detected_objects
        if confidence >= CONFIDENCE
    ]
    
    if SAVE_OUTPUT:
        output = distance.create_detection_image(current_disparity, bounding_boxes)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FILE), output)
    
    # Process and announce detected objects
    for x, y, w, h, label, obj_distance in bounding_boxes:
        if obj_distance < THRESHOLD:
            distance_ft = round(obj_distance * 3.281, 2)
            message = f"Warning: {label} detected at {obj_distance:.2f} meters ({distance_ft} feet)"
            print(message)
            speak_async(message)

def on_button_press(channel):
    p = multiprocessing.Process(target=button_press_action)
    p.start()

# Stereo block matching parameters
SWS = 15        # Block size (SADWindowSize)
PFS = 9         # Pre-filter size
PFC = 29        # Pre-filter cap
MDS = -30       # Minimum disparity
NOD = 16 * 9    # Number of disparities
TTH = 100       # Texture threshold
UR = 10         # Uniqueness ratio
SR = 14         # Speckle range
SPWS = 100      # Speckle window size

# Initialize cameras
camera_left = initialize_camera(0, img_width, img_height)
camera_right = initialize_camera(1, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

# Load calibration
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder=CALIB_RESULTS)

# Initialize interface windows
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.moveWindow("Disparity", 50, 100)
cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.moveWindow("Left", 450, 100)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.moveWindow("Right", 850, 100)

# Initialize StereoBM
sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)

# Audio process handling
audio_process = None
def speak_async(text):
    global audio_process
    if audio_process is None or not audio_process.is_alive():
        audio_process = multiprocessing.Process(target=speak, args=(text, 3, 90))
        audio_process.start()

def stereo_depth_map(rectified_pair, baseline, focal_length, bounding_boxes=None):
    dmLeft, dmRight = rectified_pair
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
    local_max = disparity.max()
    local_min = disparity.min()

    # Normalize and visualize the disparity map
    disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    
    # Overlay bounding boxes if available
    if bounding_boxes:
        for x, y, w, h, label, obj_distance in bounding_boxes:
            cv2.rectangle(disparity_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} ({obj_distance:.2f}m)"
            cv2.putText(disparity_color, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Disparity", disparity_color)
    return disparity_color, disparity

def load_map_settings(fName):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    print('Loading parameters from file...')
    with open(fName, 'r') as f:
        data = json.load(f)
        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']

        # Apply settings to StereoBM object
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(PFS)
        sbm.setPreFilterCap(PFC)
        sbm.setMinDisparity(MDS)
        sbm.setNumDisparities(NOD)
        sbm.setTextureThreshold(TTH)
        sbm.setUniquenessRatio(UR)
        sbm.setSpeckleRange(SR)
        sbm.setSpeckleWindowSize(SPWS)
    
    print('Parameters loaded from file ' + fName)

# Load stereo mapping settings
load_map_settings(SETTINGS_FILE)

# Attach button press handler
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=on_button_press, bouncetime=1000)

try:
    # Main loop - continuous stereo vision
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
        disparity_color, current_disparity = stereo_depth_map(rectified_pair, BASELINE, focal_length_px)
        
        # Show the rectified images
        cv2.imshow("Left", rectified_pair[0])
        cv2.imshow("Right", rectified_pair[1])
        
        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    # Cleanup
    GPIO.cleanup()
    camera_left.stop()
    camera_right.stop()
    cv2.destroyAllWindows()
