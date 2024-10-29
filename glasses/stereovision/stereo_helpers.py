import json
import cv2
import numpy as np
from math import tan, pi

def load_map_settings(fName, sbm):
    """Load and apply stereo block matching parameters from a configuration file."""
    print('Loading parameters from file...')
    with open(fName, 'r') as f:
        data = json.load(f)
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(data['preFilterSize'])
        sbm.setPreFilterCap(data['preFilterCap'])
        sbm.setMinDisparity(data['minDisparity'])
        sbm.setNumDisparities(data['numberOfDisparities'])
        sbm.setTextureThreshold(data['textureThreshold'])
        sbm.setUniquenessRatio(data['uniquenessRatio'])
        sbm.setSpeckleRange(data['speckleRange'])
        sbm.setSpeckleWindowSize(data['speckleWindowSize'])
    
    print('Parameters loaded from file ' + fName)

def stereo_disparity_map(rectified_pair, sbm):
    """Compute the disparity map for the rectified stereo images."""
    dmLeft, dmRight = rectified_pair
    disparity = sbm.compute(dmLeft, dmRight)
    return disparity

def calculate_distance(disparity, baseline, focal_length):
    """Calculate the distance map from a given disparity map."""
    # Avoid division by zero in disparity by replacing zero values with np.inf
    disparity[disparity == 0] = np.inf
    # Calculate the distance map using the baseline and focal length
    distance_map = (focal_length * baseline) / disparity
    return distance_map

def before_threshold(depth_map, threshold):
    """Return a boolean mask where values in the depth map are smaller than the specified threshold."""
    mask = depth_map < threshold
    return mask


def stereo_depth_map(disparity, baseline, focal_length, threshold):
    """Process the disparity map into a depth map and apply a threshold mask."""
    # Calculate depth map from disparity
    depth_map = calculate_distance(disparity, baseline, focal_length)
    # Apply threshold to the depth map
    thresholded_map = before_threshold(depth_map, threshold)
    
    return (depth_map, thresholded_map)
