"""
For rectifying stereo images and computing disparity maps.
"""
import cv2
import numpy as np

def rectify_imgs(left, right, calibration_dir="../data/stereo_images/scenes/calibration_results"):
    # Step 1: Load Calibration Data and Rectify Images
    # Load calibration data
    left_npzfile = np.load("{}/calibration_left.npz".format(calibration_dir))
    right_npzfile = np.load("{}/calibration_right.npz".format(calibration_dir))
    stereo_npzfile = np.load("{}/stereo_calibration.npz".format(calibration_dir))
   
    # rectification maps
    left_map_x_rectify = stereo_npzfile["left_map_x_rectify"]
    left_map_y_rectify = stereo_npzfile["left_map_y_rectify"]
    right_map_x_rectify = stereo_npzfile["right_map_x_rectify"]
    right_map_y_rectify = stereo_npzfile["right_map_y_rectify"]
    
    Q = stereo_npzfile["disparityToDepthMap"]
    
    focal_length = left_npzfile['camera_matrix'][0][0]

    # Apply rectification maps (from calibration results) to get rectified images
    left_rectified = cv2.remap(left, left_map_x_rectify, left_map_y_rectify, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, right_map_x_rectify, right_map_y_rectify, interpolation=cv2.INTER_LINEAR)
    
    return left_rectified, right_rectified, Q, focal_length

def make_disparity_map(left_rectified, right_rectified, min_disp=0, num_disp=16*9, block_size=9):
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2
    )

    # Compute disparity map
    disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    
    return disparity_map