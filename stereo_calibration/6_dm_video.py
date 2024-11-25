import time, os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibration
from datetime import datetime
from math import tan, pi
from camera.cam_config import initialize_camera
from tuning_helper import load_map_settings_with_sbm


THRESHOLD = 2.5   # Threshold in meters (2.5m)
CONFIG_FILE = "./cam_config.json"
SETTINGS_FILE = "./3dmap_set.txt"
CALIB_RESULTS = './calib_result'

# Load configuration from config.json
config_path = CONFIG_FILE
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
# Camera calibration parameters
BASELINE = int(config['baseline_length_mm']) / 1000  # Baseline in m (60mm)
FOCAL = int(config['focal_length_mm']) / 1000        # Focal length in m (2.6mm)
SENSOR_WIDTH = float(config['cmos_size_m'])          # Sensor width in m (1/4in)
H_FOV = int(config['field_of_view']['horizontal'])   # Horizontal field of view (73deg)
scale_ratio = float(config['scale_ratio'])           # Image scaling ratio (0.5)
cam_width = int(config['image_width'])
cam_height = int(config['image_height'])

# Camera resolution height must be divisible by 16, and width by 32
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Image resolution: " + str(img_width) + " x " + str(img_height))

# Focal length in pixels
focal_length_px = (img_width * 0.5) / tan(H_FOV * 0.5 * pi / 180) # Focal length in px with FOV
print("Focal length: " + str(focal_length_px) + " px")

# Initialize the cameras
camera_left = initialize_camera(1, img_width, img_height)
camera_right = initialize_camera(0, img_width, img_height)

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
sbm = load_map_settings_with_sbm(SETTINGS_FILE)

# Function to calculate the depth (distance) of the center pixel
def calculate_distance(disparity, baseline, focal_length):
    # Get the center pixel coordinates
    normalized_disparity = disparity + 61.0
    center_x = normalized_disparity.shape[1] // 2
    center_y = normalized_disparity.shape[0] // 2
    center_disparity = normalized_disparity[center_y, center_x]
    
    if center_disparity > 0:  # Ensure disparity is positive
        # Calculate the distance Z
        distance = (focal_length * baseline) / center_disparity
        return distance
    else:
        return float('inf')  # Infinite distance if disparity is zero or negative

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
    center_distance = calculate_distance(disparity, baseline, focal_length)
    center_distance = round(center_distance, 4)
    thresh_ft = round(THRESHOLD * 3.281, 3)
    dist_ft = round(center_distance * 3.281, 3)
    if center_distance < THRESHOLD:
        print(f"Threshhold({THRESHOLD}m={thresh_ft}ft) breached, center: {center_distance}m = {dist_ft}ft")
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        quit()
    return disparity_color

# Capture frames from the camera continuously
while True:
    frame_left = camera_left.capture_array()
    frame_right = camera_right.capture_array()

    # Convert to grayscale
    imgLeft = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    imgRight = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    
    # Rectify the stereo pair using calibration data
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    
    # Generate and display the depth map, and calculate center distance
    disparity = stereo_depth_map(rectified_pair, BASELINE, focal_length_px)
    
    # Show the left and right images
    cv2.imshow("left", imgLeft)
    cv2.imshow("right", imgRight)
