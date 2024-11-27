"""
This file is used to test the stereo camera setup.
It captures frames from the stereo camera and displays them in a single window.
"""
import time
import cv2
import numpy as np
import os, json
from datetime import datetime
from camera.cam_config import initialize_camera

# File for captured image
config_path = "cam_config.json"

if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
# Camera settings
cam_width = config["image_width"]
cam_height = config["image_height"]
scale_ratio = config["scale_ratio"]

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
camera_left = initialize_camera(1, img_width, img_height)
camera_right = initialize_camera(0, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

t2 = datetime.now()
counter = 0
avgtime = 0

# Capture frames from the camera
while True:
    frameL = camera_left.capture_array()
    frameR = camera_right.capture_array()
    
    # Concatenate frames horizontally
    frame = np.hstack((frameL, frameR))
    
    counter += 1
    t1 = datetime.now()
    timediff = t1-t2
    avgtime = avgtime + (timediff.total_seconds())
    
    cv2.imshow("Stereo Pair", frame)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        avgtime = avgtime/counter
        print ("Average time between frames: " + str(avgtime))
        print ("Average FPS: " + str(1/avgtime))
        break

# Cleanup
camera_left.stop()
camera_right.stop()
cv2.destroyAllWindows()
