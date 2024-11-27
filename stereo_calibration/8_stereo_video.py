import json
import os
import time
import cv2
import numpy as np
from rectify import rectify_imgs, make_disparity_map
from camera.cam_config import initialize_camera

# Directories and Constants
CALIBRATION_DIR = '../data/stereo_images/scenes/calibration_results'
BASELINE = 0.06
DISTANCE_FACTOR = BASELINE * 5 / 8  # (25 / 39)

# Load configuration from config.json
config_path = "cam_config.json"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Camera settings
cam_width = config["image_width"]
cam_height = config["image_height"]
scale_ratio = config["scale_ratio"]

# Camera resolution height must be divisible by 16, and width by 32
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
print("Scaled image resolution: " + str(img_width) + " x " + str(img_height))

# Initialize the cameras
camera_left = initialize_camera(1, img_width, img_height)
camera_right = initialize_camera(0, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

print("Starting stereo camera depth estimation...")

# Capture frames from the cameras
while True:
    frameL = camera_left.capture_array()
    frameR = camera_right.capture_array()

    # Rectify the images
    left_rectified, right_rectified, Q, focal_length = rectify_imgs(frameL, frameR, CALIBRATION_DIR)

    # Generate the disparity map
    min_disp = 0
    num_disp = 16 * 2  # must be divisible by 16
    block_size = 10
    disparity_map = make_disparity_map(left_rectified, right_rectified, min_disp, num_disp, block_size)

    # Convert disparity to depth map
    depth_map = cv2.reprojectImageTo3D(disparity_map, Q)

    # Calculate the distance of the center pixel
    center_x, center_y = depth_map.shape[1] // 2, depth_map.shape[0] // 2
    center_distance = depth_map[center_y, center_x, 2] * DISTANCE_FACTOR
    print(f"Distance at center pixel ({center_x}, {center_y}): {center_distance:.2f} meters")

    # Draw a dot at the center pixel on the rectified left image
    cv2.circle(left_rectified, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot with radius 5

    # Display the depth map
    depth_display = depth_map[:, :, 2] * DISTANCE_FACTOR  # Z values represent depth
    depth_display_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_display, alpha=255.0 / np.max(depth_display)), cv2.COLORMAP_JET)

    cv2.imshow("Rectified Left Image", left_rectified)
    cv2.imshow("Depth Map (in meters)", depth_display_colored)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
camera_left.stop()
camera_right.stop()
cv2.destroyAllWindows()
