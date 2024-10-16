import sys, os, time
import numpy as np
import cv2

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    """Find the depth of a point in 3D space.
    
    Args:
        right_point: The point in the right image.
        left_point: The point in the left image.
        frame_right: The right image.
        frame_left: The left image.
        baseline: The baseline distance between the two cameras.
        f: The focal length of the cameras.
        alpha: The skew angle of the cameras.
        
    Returns:
        The depth of the point in 3D space.
    """
    height_right, width_right, depth_right, = frame_right.shape
    height_left, width_left, depth_left, = frame_left.shape
    
    # Calculate the focal length of the cameras
    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print("Image dimensions are not equal.")
    
    # Calculate distance to point
    x_right = right_point[0]
    x_left = left_point[0]
    disparity = x_right - x_left
    zDepth = (f_pixel * baseline) / disparity
    
    return abs(zDepth)

def compute_disparity_map(frame_right, frame_left):
    """Compute the disparity map for the stereo image pair."""
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape
    
    # Check that the dimensions match
    if width_right != width_left or height_right != height_left:
        print("Image dimensions do not match.")
        return None
    
    # Convert the images to grayscale for disparity computation
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    
    # Use StereoSGBM to compute the disparity map with adjusted parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 6,  # Increase to cover larger depth range
        blockSize=11,  # Increase block size to reduce noise
        P1=8 * 3 * 11 ** 2,
        P2=32 * 3 * 11 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,  # Increase to filter out bad matches
        speckleWindowSize=200,  # Increase to reduce speckle noise
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute the disparity map
    disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Avoid division by zero
    disparity_map[disparity_map == 0] = 0.1
    disparity_map[disparity_map == -1] = 0.1

    # Apply a bilateral filter to smooth disparity map
    disparity_map_filtered = cv2.bilateralFilter(disparity_map, 9, 75, 75)
    
    return disparity_map_filtered


def compute_depth_map(disparity_map, baseline, f_pixel):
    """Compute the depth map using the disparity map.
    
    Args:
        disparity_map: The disparity map of the scene.
        baseline: The baseline distance between the two cameras.
        f_pixel: The focal length of the cameras in pixels.
        
    Returns:
        The depth map of the scene.
    """
    # Compute the depth map using the disparity map
    depth_map = (f_pixel * baseline) / disparity_map
    
    return depth_map
    
