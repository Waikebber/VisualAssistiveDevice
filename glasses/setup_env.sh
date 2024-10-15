#!/bin/bash

# install necessary librariesy
sudo apt install -y python3-venv libcamera-apps libcamera-dev python3-libcamera python3-kms++ python3-picamera2 python3-pillow

# Create the virtual environment
python3 -m venv --system-site-packages yolo-env

# Activate the virtual environment
source yolo-env/bin/activate

# Install the required Python packages via pip (only those not available via apt)
echo "Installing OpenCV (cv2)..."
pip install opencv-python-headless  # Use headless if no display needed

echo "Installing YOLO (ultralytics)..."
pip install ultralytics

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

echo "Setup complete. Virtual environment created and packages installed."
echo "Activate venv with 'source yolo-env/bin/activate'"
