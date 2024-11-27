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

import os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# Load configuration from config.json
config_path = "cam_config.json"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
# Global variables preset
total_photos = int(config['total_photos'])
scale_ratio = float(config['scale_ratio'])
img_width =  int(int(config['image_width']) * scale_ratio)
img_height = int(int(config['image_height']) * scale_ratio) 
photo_width = img_width * 2 
photo_height = img_height
image_size = (img_width, img_height)

# Scaling factor for display window
display_scale = 0.5  # Adjust this value to fit the window on your screen

# Chessboard parameters
rows = int(config['chess_rows'])             # Number of rows in the chessboard
columns = int(config['chess_columns'])       # Number of columns in the chessboard
square_size = float(config['chess_square_cm']) # Size of the chessboard square in cm
print(f"Chessboard Square Size: {square_size}cm")

# Initialize the StereoCalibrator with the new resolution and chessboard size
calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 0

print('Start cycle')

while photo_counter != total_photos:
    photo_counter += 1
    print('Import pair No ' + str(photo_counter))
    
    leftName = './pairs/left_' + str(photo_counter).zfill(2) + '.png'
    rightName = './pairs/right_' + str(photo_counter).zfill(2) + '.png'
    
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
calibration.export('calib_result')
print('Calibration complete!')

average_error = calibrator.check_calibration(calibration)
print(average_error)

# Rectify and show the last pair after calibration
calibration = StereoCalibration(input_folder='calib_result')
rectified_pair = calibration.rectify((imgLeft, imgRight))

# Resize the rectified images for display
rectified_left_display = cv2.resize(rectified_pair[0], (0, 0), fx=display_scale, fy=display_scale)
rectified_right_display = cv2.resize(rectified_pair[1], (0, 0), fx=display_scale, fy=display_scale)

# Display the rectified images
cv2.imshow('Left CALIBRATED', rectified_left_display)
cv2.imshow('Right CALIBRATED', rectified_right_display)

# Save the rectified images
cv2.imwrite("rectified_left.jpg", rectified_pair[0])
cv2.imwrite("rectified_right.jpg", rectified_pair[1])

cv2.waitKey(0)
cv2.destroyAllWindows()
