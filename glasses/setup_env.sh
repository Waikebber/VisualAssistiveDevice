#!/bin/bash

# Update and install python3-venv if not already installed
sudo apt install -y python3-venv libcamera-apps libcamera-dev python3-libcamera python3-kms++ python3-picamera2

# Create the virtual environment
python3 -m venv yolo-env

# Activate the virtual environment
source yolo-env/bin/activate

# Install the required packages
echo "Installing OpenCV (cv2)..."
pip install opencv-python

echo "Installing YOLO (ultralytics)..."
pip install ultralytics

echo "Installing Picamera2..."
sudo apt install -y libcamera-apps libcamera-dev
pip install picamera2

echo "Installing Pillow..."
pip install Pillow

# Verify installation by running a simple Python script
python3 -c "
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
from PIL import Image
print('All packages installed successfully!')
"

# Deactivate the virtual environment
deactivate
python3 -m venv --system-site-packages yolo-env

echo "Setup complete. Virtual environment created and packages installed."
echo "Activate venv with 'source yolo-env/bin/activate'"
