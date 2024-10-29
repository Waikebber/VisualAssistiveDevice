# This file is a modified version of Eugene Pomazov's code from the StereoPi tutorial scripts.
# The original code can be found at: https://github.com/realizator/stereopi-tutorial
# Copyright (C) 2019 Eugene Pomazov
# Modified by Kai Webber on 10/29/2024

from picamera import PiCamera
import time, os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime
from math import tan, pi

# Load configuration from config.json
config_path = "./cam_config.json"
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

# Distance threshold for distance limit
THRESHOLD = 1    # Threshold in meters (1m)

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


# Camera resolution height must be divisible by 16, and width by 32
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Scaled image resolution: " + str(img_width) + " x " + str(img_height))

# Focal length in pixels
# focal_length_px = (FOCAL / SENSOR_WIDTH) * img_width  # Focal length in px without FOV
focal_length_px = (img_width * 0.5) / tan(H_FOV * 0.5 * pi / 180) # Focal length in px with FOV
print("Focal length: " + str(focal_length_px) + " px")

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = 20
camera.hflip = True

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50, 100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450, 100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850, 100)

# Initialize the StereoBM (Block Matching) object with updated parameters
sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)

# Function to calculate the depth (distance) of the center pixel
def calculate_distance(disparity, baseline, focal_length):
    # Get the center pixel coordinates
    center_x = disparity.shape[1] // 2
    center_y = disparity.shape[0] // 2
    center_disparity = disparity[center_y, center_x]
    
    if center_disparity > 0:  # Ensure disparity is positive
        # Calculate the distance Z
        distance = (focal_length * baseline) / center_disparity
        return distance
    else:
        return float('inf')  # Infinite distance if disparity is zero or negative

def stereo_depth_map(rectified_pair, baseline, focal_length):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
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

load_map_settings("./configCamera/3dmap_set.txt")

# Capture frames from the camera continuously
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width, img_height)):
    t1 = datetime.now()
    
    # Convert to grayscale and split stereo pair
    pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img[0:img_height, 0:int(img_width / 2)]  # Left image
    imgRight = pair_img[0:img_height, int(img_width / 2):img_width]  # Right image
    
    # Rectify the stereo pair using calibration data
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    
    # Generate and display the depth map, and calculate center distance
    disparity = stereo_depth_map(rectified_pair, BASELINE, focal_length_px)
    
    # Show the left and right images
    cv2.imshow("left", imgLeft)
    cv2.imshow("right", imgRight)

    t2 = datetime.now()
    # print("DM build time: " + str(t2 - t1))
