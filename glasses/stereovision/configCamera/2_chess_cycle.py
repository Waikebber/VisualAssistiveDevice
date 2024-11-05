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
# This file is a modified version of Eugene Pomazov's code from the StereoPi tutorial scripts.
# The original code can be found at: https://github.com/realizator/stereopi-tutorial
# Copyright (C) 2019 Eugene Pomazov
# Modified by Kai Webber on 10/29/2024

import os, time, json
from datetime import datetime
import picamera
from picamera import PiCamera
import cv2
import numpy as np

# Load Camera settings
config_path = "/data/stereo/cam_config.json"
scene_path = "/data/stereo/scenes"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
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
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Scaled image resolution: " + str(img_width) + " x " + str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = 20  # Adjust based on system performance at higher resolution
camera.hflip = True

# Start taking photos! 
counter = 0
t2 = datetime.now()
print("Starting photo sequence")
for frame in camera.capture_continuous(capture, format="bgra", \
                  use_video_port=True, resize=(img_width, img_height)):
    t1 = datetime.now()
    cntdwn_timer = countdown - int((t1 - t2).total_seconds())
    
    # If countdown is zero - record the next image
    if cntdwn_timer == -1:
        counter += 1
        filename = scene_path+ '/scene_' + str(img_width) + 'x' + str(img_height) + '_' + \
                   str(counter) + '.png'
        cv2.imwrite(filename, frame)
        print(' [' + str(counter) + ' of ' + str(total_photos) + '] ' + filename)
        t2 = datetime.now()
        time.sleep(1)
        cntdwn_timer = 0  # To avoid "-1" timer display
        next

    # Draw countdown counter, seconds
    cv2.putText(frame, str(cntdwn_timer), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'Q' key to quit, or wait till all photos are taken
    if (key == ord("q")) or (counter == total_photos):
        break

print("Photo sequence finished")

