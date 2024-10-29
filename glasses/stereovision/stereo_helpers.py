import json
import cv2
import numpy as np
from math import tan, pi

def calculate_center_distance(disparity, baseline, focal_length):
    """Calculate the distance of the center pixel from the stereo disparity map."""
    center_x = disparity.shape[1] // 2
    center_y = disparity.shape[0] // 2
    center_disparity = disparity[center_y, center_x]
    
    if center_disparity > 0:
        distance = (focal_length * baseline) / center_disparity
        return distance
    else:
        return float('inf')

def stereo_depth_map(rectified_pair, baseline, focal_length, sbm, threshold):
    """Generate a depth map from rectified stereo images and calculate center distance."""
    dmLeft, dmRight = rectified_pair
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()

    disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    
    center_distance = calculate_center_distance(disparity, baseline, focal_length)
    center_distance = round(center_distance, 4)
    thresh_ft = round(threshold * 3.281, 3)
    dist_ft = round(center_distance * 3.281, 3)
    if center_distance < threshold:
        print(f"Threshold({threshold}m={thresh_ft}ft) breached, center: {center_distance}m = {dist_ft}ft")
    
    return disparity_color

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
