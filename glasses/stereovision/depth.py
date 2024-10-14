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
    
