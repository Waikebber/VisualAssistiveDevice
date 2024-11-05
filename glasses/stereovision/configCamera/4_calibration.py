# Copyright (C) 2019 Eugene Pomazov, <stereopi.com>, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
# Most of this code is updated version of 3dberry.org project by virt2real
# 
# Thanks to Adrian and http://pyimagesearch.com, as there are lot of
# code in this tutorial was taken from his lessons.
# 
# This file is a modified version of Eugene Pomazov's code from the StereoPi tutorial scripts.
# The original code can be found at: https://github.com/realizator/stereopi-tutorial
# Copyright (C) 2019 Eugene Pomazov
# Modified by Kai Webber on 10/29/2024

import os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# Load configuration from config.json
config_path = "/data/stereo/cam_config.json"
pairs_path = "/data/stereo/pairs"
calib_result = "/data/stereo/calib_result"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
# Global variables preset
total_photos = int(config['total_photos'])
scale_ratio = float(config['scale_ratio'])
photo_width =  int(int(config['image_width']) * scale_ratio)
photo_height = int(int(config['image_height']) * scale_ratio) 
img_width = photo_width // 2 
img_height = photo_height
image_size = (img_width, img_height)

# Chessboard parameters
rows = int(config['chess_rows'])             # Number of rows in the chessboard
columns = int(config['chess_columns'])       # Number of columns in the chessboard
square_size = int(config['chess_square_cm']) # Size of the chessboard square in cm

# Initialize the StereoCalibrator with the new resolution and chessboard size
calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 0

print('Start cycle')

while photo_counter != total_photos:
    photo_counter += 1
    print('Import pair No ' + str(photo_counter))
    
    leftName = pairs_path + '/left_' + str(photo_counter).zfill(2) + '.png'
    rightName = pairs_path + '/right_' + str(photo_counter).zfill(2) + '.png'
    
    # Check if both left and right images exist
    if os.path.isfile(leftName) and os.path.isfile(rightName):
        imgLeft = cv2.imread(leftName, 1)
        imgRight = cv2.imread(rightName, 1)
        
        try:
            # Try to find the chessboard corners in both images
            calibrator._get_corners(imgLeft)
            calibrator._get_corners(imgRight)
        except ChessboardNotFoundError as error:
            print(error)
            print("Pair No " + str(photo_counter) + " ignored")
        else:
            # If successful, add corners to the calibration process
            calibrator.add_corners((imgLeft, imgRight), True)
        
print('End cycle')

print('Starting calibration... It can take several minutes!')
# Calibrate the stereo camera system
calibration = calibrator.calibrate_cameras()
# Export the calibration results to the 'calib_result' folder
calibration.export(calib_result)
print('Calibration complete!')

# Rectify and show the last pair after calibration
calibration = StereoCalibration(input_folder=calib_result)
rectified_pair = calibration.rectify((imgLeft, imgRight))

# Display the rectified images
cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
# Save the rectified images
cv2.imwrite("rectified_left.jpg", rectified_pair[0])
cv2.imwrite("rectified_right.jpg", rectified_pair[1])
cv2.waitKey(0)
