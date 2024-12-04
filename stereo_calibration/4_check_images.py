import os, json
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.exceptions import ChessboardNotFoundError

OUTPUT_DIR = '../data/stereo_images/midnight/'

# Scaling factor for display window
display_scale = 1  # Adjust this value to fit the window on your screen

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

# Chessboard parameters
rows = int(config['chess_rows'])             # Number of rows in the chessboard
columns = int(config['chess_columns'])       # Number of columns in the chessboard
square_size = float(config['chess_square_cm']) # Size of the chessboard square in cm
print(f"Chessboard Square Size: {square_size}cm")
print(f"Chessboard Pattern: {rows}x{columns}")

# Initialize the StereoCalibrator with the new resolution and chessboard size
calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 0

print('Start cycle')

def verify_img(image, image_name):
    """Diagnostics for image details"""
    if image is None:
        print(f"ERROR: {image_name} is None - image could not be read")
        return False
    
    # Try manual chessboard detection
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        return ret
    except Exception as e:
        print(f"  Chessboard detection error: {e}")
        return False

while photo_counter != total_photos:
    photo_counter += 1
    print('\n' + '='*50)
    print(f'Import pair No {photo_counter}')
    
    leftName = f'{OUTPUT_DIR}left/{str(photo_counter)}.png'
    rightName = f'{OUTPUT_DIR}right/{str(photo_counter)}.png'
    rawName = f'{OUTPUT_DIR}raw/{str(photo_counter)}.png'
    
    # Check if both left and right images exist
    if os.path.isfile(leftName) and os.path.isfile(rightName):
        # Read images in color mode
        imgLeft = cv2.imread(leftName, 1)
        imgRight = cv2.imread(rightName, 1)
        
        # Detailed image diagnostics
        left_valid = verify_img(imgLeft, "Left Image")
        right_valid = verify_img(imgRight, "Right Image")
        
        if not left_valid or  not right_valid:
            print(f"Pair No {photo_counter} removed due to image reading issues")
            # Delete invalid images
            os.remove(leftName)
            os.remove(rightName)
            if os.path.isfile(rawName):
                os.remove(rawName)
            continue
        
        if not (left_valid and right_valid):
            print(f"Pair No {photo_counter} ignored due to image reading issues")
            # Delete invalid images
            os.remove(leftName)
            os.remove(rightName)
            if os.path.isfile(rawName):
                os.remove(rawName)
            continue
        
        try:
            # Attempt to get corners from both images
            corners_left = calibrator._get_corners(imgLeft)
            corners_right = calibrator._get_corners(imgRight)
            
            # Draw chessboard corners
            grayLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
            grayRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
            
            ret_left, corners_left_cv = cv2.findChessboardCorners(grayLeft, (columns, rows), None)
            ret_right, corners_right_cv = cv2.findChessboardCorners(grayRight, (columns, rows), None)
            
            cv2.drawChessboardCorners(imgLeft, (columns, rows), corners_left_cv, ret_left)
            cv2.drawChessboardCorners(imgRight, (columns, rows), corners_right_cv, ret_right)
            
            # Display images with corners
            combined_image = np.hstack((imgLeft, imgRight))
            resized_image = cv2.resize(combined_image, (int(photo_width * display_scale), int(photo_height * display_scale)))
            cv2.imshow("Stereo Pair with Chessboard - Press 'Enter' to keep, 'Backspace' to delete", resized_image)
            
            # Wait for key press
            while True:
                key = cv2.waitKey(0)
                
                if key == 13 or key == ord('\r'):  # Enter key
                    calibrator.add_corners((imgLeft, imgRight), True)
                    print("Pair No " + str(photo_counter) + " kept")
                    break
                elif key == 8 or key == ord('\b'):  # Backspace key
                    print("Pair No " + str(photo_counter) + " ignored - User deleted")
                    os.remove(leftName)
                    os.remove(rightName)
                    if os.path.isfile(rawName):
                        os.remove(rawName)
                    break
                elif key == 27:  # ESC key to exit the entire process
                    print("Process interrupted by user")
                    cv2.destroyAllWindows()
                    exit()
                else:
                    print(f"Pressed key code: {key}. Press Enter to keep, Backspace to delete.")
            
        except ChessboardNotFoundError as error:
            continue

print('End cycle')

# Cleanup
cv2.destroyAllWindows()
