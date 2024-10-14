import sys, os
import cv2
import numpy as np
from depth import compute_depth_map
from CameraCalibration.calibration import undistort_rectify

## CONSTANTS
BASELINE = 0.1 # Baseline distance between the two cameras (meters)
F = 700        # Focal length of the cameras
ALPHA = 90     # Skew angle of the cameras

IMG_LEFT = 'left_image.png'
IMG_RIGHT = 'right_image.png'


def main():
    # Load stereo images
    frame_left = cv2.imread(IMG_LEFT)
    frame_right = cv2.imread(IMG_RIGHT)
    # Undistort and rectify the images
    undistortedR, undistortedL = undistort_rectify(frame_right, frame_left)
    
    
    # Compute depth map
    depth_map = compute_depth_map(undistortedR, undistortedL, BASELINE, F, ALPHA)
    
    # Visualize the depth map
    depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_vis = np.uint8(depth_map_vis)
    
    cv2.imshow('Depth Map', depth_map_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Entry point
if __name__ == '__main__':
    main()
