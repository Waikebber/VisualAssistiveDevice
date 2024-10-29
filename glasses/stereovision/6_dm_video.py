# This file is a modified version of Eugene Pomazov's code from the StereoPi tutorial scripts.
# The original code can be found at: https://github.com/realizator/stereopi-tutorial
# Copyright (C) 2019 Eugene Pomazov
# Modified by Kai Webber on 10/29/2024
# Main script: stereo_capture.py
# This file is a modified version of Eugene Pomazov's code from the StereoPi tutorial scripts.
# The original code can be found at: https://github.com/realizator/stereopi-tutorial
# Modified by Kai Webber on 10/29/2024

from picamera import PiCamera
import time, os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibration
from datetime import datetime
from math import tan, pi

# Import the helper functions
from stereo_helpers import load_map_settings, stereo_disparity_map, calculate_distance

# Load configuration from config.json
config_path = "./cam_config.json"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
# Camera calibration parameters
BASELINE = int(config['baseline_length_mm']) / 1000  # Baseline in meters (60mm)
FOCAL = int(config['focal_length_mm']) / 1000        # Focal length in meters (2.6mm)
H_FOV = int(config['field_of_view']['horizontal'])   # Horizontal field of view (73 degrees)
scale_ratio = float(config['scale_ratio'])           # Image scaling ratio (0.5)
cam_width = int(config['image_width'])
cam_height = int(config['image_height'])

# Stereo block matching parameters
SWS = 15       # Block size (SADWindowSize) for stereo matching
PFS = 9        # Pre-filter size to smooth image noise
PFC = 29       # Pre-filter cap (intensity normalization)
MDS = -30      # Minimum disparity for close-range depth calculation
NOD = 16 * 9   # Number of disparities (multiple of 16)
TTH = 100      # Texture threshold for disparity computation
UR = 10        # Uniqueness ratio to filter ambiguous matches
SR = 14        # Speckle range to suppress noise in disparity map
SPWS = 100     # Speckle window size for disparity filtering

# Adjust camera resolution to be divisible by 16 and 32
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Scaled image resolution: " + str(img_width) + " x " + str(img_height))

# Focal length in pixels, based on field of view
focal_length_px = (img_width * 0.5) / tan(H_FOV * 0.5 * pi / 180)
print("Focal length: " + str(focal_length_px) + " px")

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = 20
camera.hflip = True

# Load calibration data for rectification
print('Reading calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')

# Initialize display windows
cv2.namedWindow("Depth Map")
cv2.moveWindow("Depth Map", 50, 100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450, 100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850, 100)

# Initialize the StereoBM (Block Matching) object
sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)

# Load stereo matching parameters from configuration file
load_map_settings("./configCamera/3dmap_set.txt", sbm)

# Capture frames from the camera continuously
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width, img_height)):
    t1 = datetime.now()
    
    # Convert to grayscale and split stereo pair
    pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img[0:img_height, 0:int(img_width / 2)]  # Left image
    imgRight = pair_img[0:img_height, int(img_width / 2):img_width]  # Right image
    
    # Rectify the stereo pair using calibration data
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    
    # Generate the disparity map
    disparity = stereo_disparity_map(rectified_pair, sbm)
    
    # Calculate the depth map from disparity
    depth_map, threshold_map = calculate_distance(disparity, BASELINE, focal_length_px)
    
    # Get the center pixel's coordinates
    center_x = depth_map.shape[1] // 2
    center_y = depth_map.shape[0] // 2
    center_distance = depth_map[center_y, center_x]
    
    # Print the distance of the center pixel
    print(f"Center pixel distance: {center_distance:.4f} meters")
    
    # Display the depth map as a color map for visualization
    depth_map_visual = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255.0 / np.max(depth_map)), cv2.COLORMAP_JET)
    cv2.imshow("Depth Map", depth_map_visual)

    # Display the left and right images
    cv2.imshow("left", imgLeft)
    cv2.imshow("right", imgRight)

    t2 = datetime.now()
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
camera.close()
cv2.destroyAllWindows()
