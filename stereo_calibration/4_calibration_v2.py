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
img_width = int(int(config['image_width']) * scale_ratio)
img_height = int(int(config['image_height']) * scale_ratio) 
photo_width = img_width * 2 
photo_height = img_height
image_size = (img_width, img_height)

# Chessboard parameters
rows = int(config['chess_rows'])             # Number of rows in the chessboard
columns = int(config['chess_columns'])       # Number of columns in the chessboard
square_size = float(config['chess_square_cm']) / 100.0  # Convert square size to meters

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
            # Add corners to the calibrator directly (calibrator will handle finding corners)
            calibrator.add_corners((imgLeft, imgRight), True)
        except ChessboardNotFoundError as error:
            print(error)
            print("Pair No " + str(photo_counter) + " ignored")
        
print('End cycle')

print('Starting calibration... It can take several minutes!')
# Calibrate the stereo camera system
calibration = calibrator.calibrate_cameras()
# Export the calibration results to the 'calib_result' folder
calibration.export('calib_result')
print('Calibration complete!')

# Rectify and show the last pair after calibration
calibration = StereoCalibration(input_folder='calib_result')
rectified_pair = calibration.rectify((imgLeft, imgRight))

# Display the rectified images
cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
# Save the rectified images
cv2.imwrite("rectified_left.jpg", rectified_pair[0])
cv2.imwrite("rectified_right.jpg", rectified_pair[1])
cv2.waitKey(0)
