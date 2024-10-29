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

import cv2
import os, json

# Load configuration from config.json
config_path = "../cam_config.json"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Global variables preset
total_photos = int(config['total_photos'])
scale_ratio = float(config['scale_ratio'])
photo_width = int(int(config['image_width']) * scale_ratio)
photo_height = int(int(config['image_height']) * scale_ratio)
img_height = photo_height  
img_width = photo_width // 2
photo_counter = 0

# Main pair cut cycle
if (os.path.isdir("./pairs") == False):
    os.makedirs("./pairs")

while photo_counter != total_photos:
    photo_counter += 1
    filename = './scenes/scene_' + str(photo_width) + 'x' + str(photo_height) + \
               '_' + str(photo_counter) + '.png'
    
    if os.path.isfile(filename) == False:
        print("No file named " + filename)
        continue

    # Load the stereo image (both left and right views combined)
    pair_img = cv2.imread(filename, -1)
    
    # Display the stereo pair
    cv2.imshow("ImagePair", pair_img)
    cv2.waitKey(0)
    
    # Split the stereo pair into left and right images
    imgLeft = pair_img[0:img_height, 0:img_width]  # Left image
    imgRight = pair_img[0:img_height, img_width:photo_width]  # Right image
    
    # Save the split left and right images
    leftName = './pairs/left_' + str(photo_counter).zfill(2) + '.png'
    rightName = './pairs/right_' + str(photo_counter).zfill(2) + '.png'
    
    cv2.imwrite(leftName, imgLeft)
    cv2.imwrite(rightName, imgRight)
    
    print('Pair No ' + str(photo_counter) + ' saved.')

print('End cycle')
