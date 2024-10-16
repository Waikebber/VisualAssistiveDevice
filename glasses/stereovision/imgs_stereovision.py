import sys, os
import cv2
import numpy as np
from depth import compute_depth_map
from CameraCalibration.calibrate import undistort_rectify

## CONSTANTS
##################### CAMERA SPECS ##############################
camera_specs = {
    "CMOS_size": "1/4 inch",
    "aperture": 2.4,
    "resolution": (3280, 2464),  # width, height in pixels
    "focal_length": 2.6,  # in mm
    "field_of_view": {
        "diagonal": 83,  # in degrees
        "horizontal": 73,  # in degrees
        "vertical": 50  # in degrees
    },
    "distortion": "< 1%",
    "baseline_length": 60  # in mm
}

F = camera_specs["focal_length"]
BASELINE = camera_specs["baseline_length"]
ALPHA = camera_specs["field_of_view"]["horizontal"]
################################################################

IMG_LEFT = 'testImgs/test1/left.png'
IMG_RIGHT = 'testImgs/test1/right.png'


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
