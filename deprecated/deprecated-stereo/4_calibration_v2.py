import os
import json
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
rows = int(config['chess_rows'])  # Number of rows in the chessboard
columns = int(config['chess_columns'])  # Number of columns in the chessboard
square_size = float(config['chess_square_cm']) / 100.0  # Convert square size to meters

# Initialize the StereoCalibrator with the new resolution and chessboard size
calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 0

# Create debug directory if it doesn't exist
if not os.path.isdir("./debug_corners"):
    os.makedirs("./debug_corners")

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

        # Convert images to grayscale for corner detection
        grayL = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners in both images
        foundL, cornersL = cv2.findChessboardCorners(grayL, (columns, rows), None)
        foundR, cornersR = cv2.findChessboardCorners(grayR, (columns, rows), None)

        if foundL and foundR:
            # Refining the corner locations for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            # Draw and save corners for visual verification
            cv2.drawChessboardCorners(imgLeft, (columns, rows), cornersL, foundL)
            cv2.drawChessboardCorners(imgRight, (columns, rows), cornersR, foundR)

            left_debug_name = f'./debug_corners/left_{photo_counter:02d}_corners.png'
            right_debug_name = f'./debug_corners/right_{photo_counter:02d}_corners.png'
            cv2.imwrite(left_debug_name, imgLeft)
            cv2.imwrite(right_debug_name, imgRight)

            # Add refined corners to the calibrator
            calibrator.add_corners((imgLeft, imgRight), True)

            # Display the images with drawn corners for visual verification
            cv2.imshow("Corners - Left", imgLeft)
            cv2.imshow("Corners - Right", imgRight)
            cv2.waitKey(500)  # Display each pair for 500 ms
        else:
            print(f"Chessboard not found in pair {photo_counter}. Skipping this pair.")
    else:
        print(f"Image pair {photo_counter} is missing. Skipping this pair.")

cv2.destroyAllWindows()  # Close any open windows after the loop
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
cv2.destroyAllWindows()
