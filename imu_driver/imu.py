import json
import os
import math
from .utils import read_sensor_data, apply_calibration, load_calibration

# Path to the calibration file
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

# Load calibration data at module load time
calibration = load_calibration(CALIBRATION_FILE)
offsets, scales = calibration["offsets"], calibration["scales"]

def distanceToGround(user_height):
    """
    Calculate the distance to the ground based on the user's height
    and calibrated IMU data.
    
    :param user_height: Height of the user in meters.
    :return: Distance to the ground in meters.
    """
    # Get raw IMU data
    raw_data = read_sensor_data()

    # Apply calibration to raw data
    calibrated_data = apply_calibration(raw_data, offsets, scales)

    # Extract calibrated accelerometer data
    accel_x, accel_y, accel_z = calibrated_data

    # Calculate pitch angle (tilt forward/backward)
    pitch = math.atan2(accel_x, math.sqrt(accel_y**2 + accel_z**2))

    # Calculate distance to the ground
    distance = user_height * math.cos(pitch)

    return distance
