from picamera2 import Picamera2
import cv2
import time

# Initialize cameras
camera_left = Picamera2(0)
camera_right = Picamera2(1)

# Create configuration matching libcamera-hello settings
config_left = camera_left.create_preview_configuration(
    main={"size": (820, 616), "format": "RGB888"},  # Halved the resolution
    raw={"size": (1640, 1232), "format": "SBGGR10_CSI2P"},
    buffer_count=4,
    controls={
        "FrameDurationLimits": (23894, 11767556),
        "NoiseReductionMode": 0,
        "AwbEnable": True,  # Enable auto white balance
        "AeEnable": True,   # Enable auto exposure
        "AnalogueGain": 1.0
    }
)

config_right = camera_right.create_preview_configuration(
    main={"size": (820, 616), "format": "RGB888"},
    raw={"size": (1640, 1232), "format": "SBGGR10_CSI2P"},
    buffer_count=4,
    controls={
        "FrameDurationLimits": (23894, 11767556),
        "NoiseReductionMode": 0,
        "AwbEnable": True,  # Enable auto white balance
        "AeEnable": True,   # Enable auto exposure
        "AnalogueGain": 1.0
    }
)

# Print configurations before applying
print("Left camera configuration:", config_left)
print("Right camera configuration:", config_right)

# Configure cameras
camera_left.configure(config_left)
camera_right.configure(config_right)

# Start cameras
camera_left.start()
camera_right.start()

# Allow time for cameras to initialize
time.sleep(2)

while True:
    # Capture frames
    frame_left = camera_left.capture_array()
    frame_right = camera_right.capture_array()
    
    # Display frames
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
camera_left.stop()
camera_right.stop()
