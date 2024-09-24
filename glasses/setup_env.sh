#!/bin/bash

# Update and install python3-venv if not already installed
sudo apt update
sudo apt install -y python3-venv

# Create the virtual environment
python3 -m venv my_project_env

# Activate the virtual environment
source my_project_env/bin/activate

# Install the required packages
echo "Installing OpenCV (cv2)..."
pip install opencv-python

echo "Installing YOLO (ultralytics)..."
pip install ultralytics

echo "Installing Picamera2..."
sudo apt install -y libcamera-apps libcamera-dev
pip install picamera2

# Verify installation by running a simple Python script
python3 -c "
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
print('All packages installed successfully!')
"

# Deactivate the virtual environment
deactivate

echo "Setup complete. Virtual environment created and packages installed."
