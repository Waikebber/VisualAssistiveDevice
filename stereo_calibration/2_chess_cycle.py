# Copyright (C) 2019 Eugene Pomazov, <stereopi.com>, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
# Most of this code is updated version of 3dberry.org project by virt2real
# 
# Thanks to Adrian and http://pyimagesearch.com, as there are lot of
# code in this tutorial was taken from his lessons.
# 
"""
This file is used to take a series of photos with a countdown timer.
It captures frames from the stereo camera and saves them to the scenes directory.
Take pictures of a chessboard pattern to calibrate the stereo camera.
"""
import os
import time
import json
from datetime import datetime
import cv2
import numpy as np
from camera.cam_config import initialize_camera

SAVE_PATH = "../data/midnight/"

# Load Camera settings
config_path = "cam_config.json"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
# Create scenes directory if it doesn't exist
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

os.makedirs(SAVE_PATH, exist_ok=True)

# Photo session settings
total_photos = int(config['total_photos'])
cam_width = config['image_width']
cam_height = config['image_height']
scale_ratio = float(config['scale_ratio'])

# Countdown timer
countdown = int(config['countdown'])       # Interval for countdown timer, seconds
font = cv2.FONT_HERSHEY_SIMPLEX  # Countdown timer font

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

# Start taking photos! 
counter = 0
t2 = datetime.now()
print("Starting photo sequence")

try:
    while True:
        # Capture frames from both cameras
        frameL = camera_left.capture_array()
        frameR = camera_right.capture_array()
        
        # Combine frames side by side
        frame = np.hstack((frameL, frameR))
        
        t1 = datetime.now()
        cntdwn_timer = countdown - int((t1 - t2).total_seconds())
        
        # If countdown is zero - record the next image
        if cntdwn_timer == -1:
            counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'scene_{img_width}x{img_height}_{counter}.png'
            filepath = os.path.join(SAVE_PATH, filename)
            cv2.imwrite(filepath, frame)
            print(f' [{counter} of {total_photos}] {filename}')
            t2 = datetime.now()
            time.sleep(1)
            cntdwn_timer = countdown  # Reset timer
            
            if counter >= total_photos:
                print("Completed all photos")
                break

        # Draw countdown counter, seconds
        cv2.putText(frame, str(max(0, cntdwn_timer)), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
        
        # Add labels to identify left and right frames
        cv2.putText(frame, "LEFT", (50, img_height - 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (img_width + 50, img_height - 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Stereo Pair", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 'Q' key to quit
        if key == ord("q"):
            break

finally:
    # Cleanup
    print("Photo sequence finished")
    camera_left.stop()
    camera_right.stop()
    cv2.destroyAllWindows()
