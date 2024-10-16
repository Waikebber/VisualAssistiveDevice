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


def save_depth_map(depth_map, output_path='depth_map.png'):
    """Save the depth map to a file."""
    depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_vis = np.uint8(depth_map_vis)
    cv2.imwrite(output_path, depth_map_vis)
    print(f"Depth map saved to {output_path}")

def save_rectified_images(left_image, right_image, output_path_left='rectified_left.png', output_path_right='rectified_right.png'):
    """Save the rectified left and right images."""
    cv2.imwrite(output_path_left, left_image)
    cv2.imwrite(output_path_right, right_image)
    print(f"Rectified left image saved to {output_path_left}")
    print(f"Rectified right image saved to {output_path_right}")


def main():
    # Load stereo images
    frame_left = cv2.imread(IMG_LEFT)
    frame_right = cv2.imread(IMG_RIGHT)

    # Check if images are loaded
    if frame_left is None or frame_right is None:
        print("Error: Could not load one or both of the images.")
        return

    # Undistort and rectify the images
    undistortedR, undistortedL = undistort_rectify(frame_right, frame_left)

    # Save rectified images
    save_rectified_images(undistortedL, undistortedR)

    # Compute depth map
    depth_map = compute_depth_map(undistortedR, undistortedL, BASELINE, F, ALPHA)

    # Save depthmap
    save_depth_map(depth_map)


# Entry point
if __name__ == '__main__':
    main()
