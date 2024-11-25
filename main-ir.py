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
        return
        
    print("Button pressed - performing object detection and distance measurement")
    
    # Convert rectified grayscale to BGR for object detection
    rectified_color = cv2.cvtColor(current_pair[0], cv2.COLOR_GRAY2BGR)
    
    detected_objects = img_recognizer.predict_frame(rectified_color)
    print('OBJECTS')
    print(detected_objects)
    
    if SAVE_OUTPUT:
        output = distance.create_detection_image(disparity_on_button_press, detected_objects)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FILE), output)
    
    # Calculate distances for detected objects using current disparity
    object_distances = distance.calculate_object_distances(disparity_on_button_press, detected_objects)
    
    # Process and announce detected objects within threshold
    for obj_name, distance, confidence, coords in object_distances:
        if distance < THRESHOLD and CONFIDENCE <= confidence:
            distance_ft = round(distance * 3.281, 2)
            message = f"Warning: {obj_name} detected at {distance:.2f} meters ({distance_ft} feet)"
            print(message)
            speak_async(message)

def on_button_press(channel):
    p = multiprocessing.Process(target=button_press_action)
    p.start()

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=on_button_press, bouncetime=200)

# Stereo block matching parameters
SWS = 15        # Block size (SADWindowSize) for stereo matching
PFS = 9         # Pre-filter size to smooth image noise
PFC = 29        # Pre-filter cap (intensity normalization)
MDS = -30       # Minimum disparity for close-range depth calculation
NOD = 16 * 9    # Number of disparities (multiple of 16)
TTH = 100       # Texture threshold for disparity computation
UR = 10         # Uniqueness ratio to filter ambiguous matches
SR = 14         # Speckle range to suppress noise in disparity map
SPWS = 100      # Speckle window size for disparity filtering

# Initialize the cameras
camera_left = initialize_camera(0, img_width, img_height)
camera_right = initialize_camera(1, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder=CALIB_RESULTS)

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50, 100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450, 100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850, 100)

# Initialize the StereoBM (Block Matching) object with updated parameters
sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)

audio_process = None
def speak_async(text):
    global audio_process
    if audio_process is None or not audio_process.is_alive():
        audio_process = multiprocessing.Process(target=speak, args=(text, 3, 90))
        audio_process.start()

def stereo_depth_map(rectified_pair, baseline, focal_length):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
    local_max = disparity.max()
    local_min = disparity.min()

    # Improved normalization and visualization of the depth map
    disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    
    # Show the depth map
    cv2.imshow("Image", disparity_color)
    
    # Calculate and print the distance of the center pixel
    center_distance = distance.calculate_center_distance(disparity)
    center_distance = round(center_distance, 4)
    thresh_ft = round(THRESHOLD * 3.281, 3)
    dist_ft = round(center_distance * 3.281, 3)
    if center_distance < THRESHOLD:
        print(f"Threshold({THRESHOLD}m={thresh_ft}ft) breached, center: {center_distance}m = {dist_ft}ft")
        speak_async(f"Threshold({THRESHOLD}m={thresh_ft}ft) breached, center: {center_distance}m = {dist_ft}ft")
    
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

        # Apply loaded parameters to StereoBM object
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

load_map_settings(SETTINGS_FILE)
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
        disparity_color, current_disparity = stereo_depth_map(rectified_pair, BASELINE, focal_length_px)
    
        # Analyze the disparity map for the closest object
        closest_distance, mean_distance, std_distance = distance.analyze_disparity_distribution(
            current_disparity
        )
    
        # Notify the user if a close object is detected
        if closest_distance < THRESHOLD:
            message = f"Closest object detected at {closest_distance:.2f} meters."
            print(message)
            speak_async(message)
    
        # Display rectified images and disparity map
        cv2.imshow("Left", rectified_pair[0])
        cv2.imshow("Right", rectified_pair[1])
        cv2.imshow("Disparity", disparity_color)
    
        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except Exception as e:
    # Log any errors that occur during the loop
    print(f"Error during processing: {e}")

finally:
    # Ensure cleanup happens no matter what
    camera_left.stop()
    camera_right.stop()
    cv2.destroyAllWindows()
