import os, json
import cv2
import numpy as np
from picamera import PiCamera
from stereovision.calibration import StereoCalibration
from math import tan, pi
from core.event_bus import EventBus
from .glasses import load_map_settings, stereo_disparity_map

DATA_PATH = "/data/stereo/"

class StereoProcessor:
    def __init__(self, event_bus: EventBus, config_path=os.path.join(DATA_PATH, "cam_config.json"), map_config_path=os.path.join(DATA_PATH, "3dmap_set.txt")):
        # Initialize EventBus and configuration
        self.event_bus = event_bus
        self.config = self.load_config(config_path)
        self.baseline = self.config['baseline_length_mm'] / 1000.0  # Convert mm to meters
        self.focal_length = self.config['focal_length_mm'] / 1000.0  # Convert mm to meters
        self.h_fov = self.config['field_of_view']['horizontal']

        # Adjust camera resolution and initialize capture
        self.cam_width, self.cam_height, self.scale_ratio = self.setup_camera_dimensions()
        self.camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
        self.camera.resolution = (self.cam_width, self.cam_height)
        self.camera.framerate = 20
        self.camera.hflip = True
        self.capture = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)

        # Focal length in pixels for depth calculation
        self.focal_length_px = (self.img_width * 0.5) / tan(self.h_fov * 0.5 * pi / 180)

        # Load stereo calibration data
        self.calibration = StereoCalibration(input_folder=os.path.join(DATA_PATH, 'calib_result'))

        # Set up StereoBM and load map settings
        self.sbm = cv2.StereoBM_create()
        load_map_settings(map_config_path, self.sbm)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def setup_camera_dimensions(self):
        cam_width = int((self.config['image_width'] + 31) / 32) * 32
        cam_height = int((self.config['image_height'] + 15) / 16) * 16
        scale_ratio = self.config['scale_ratio']
        self.img_width = int(cam_width * scale_ratio)
        self.img_height = int(cam_height * scale_ratio)
        return cam_width, cam_height, scale_ratio

    def get_current_frame(self):
        """Captures the current stereo frame and returns it as color and grayscale left/right images."""
        for frame in self.camera.capture_continuous(self.capture, format="bgra", use_video_port=True, resize=(self.img_width, self.img_height)):
            # Split stereo pair into left and right color images
            img_left_color = frame[0:self.img_height, 0:self.img_width // 2, :3]  # Left color image
            img_right_color = frame[0:self.img_height, self.img_width // 2:, :3]  # Right color image

            # Convert to grayscale for depth processing
            img_left_gray = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2GRAY)
            img_right_gray = cv2.cvtColor(img_right_color, cv2.COLOR_BGR2GRAY)
            
            return (img_left_color, img_right_color), (img_left_gray, img_right_gray)

    def calculate_disparity(self, img_left_gray, img_right_gray):
        """Rectifies and computes the disparity map from grayscale left and right images."""
        rectified_pair = self.calibration.rectify((img_left_gray, img_right_gray))
        disparity = stereo_disparity_map(rectified_pair, self.sbm)
        return disparity

    def calculate_distance(self, disparity_map, x, y):
        """Calculates the distance for a specific point (x, y) in the disparity map."""
        disparity_value = disparity_map[y, x]
        if disparity_value <= 0:
            return None  # Indicates invalid disparity
        distance = (self.focal_length_px * self.baseline) / disparity_value
        return distance

    def run_stereo_processing(self, threshold_distance=1.0, show_result=True):
        """Runs continuous stereovision processing, identifying objects within threshold."""
        while True:
            (img_left_color, _), (img_left_gray, img_right_gray) = self.get_current_frame()
            disparity_map = self.calculate_disparity(img_left_gray, img_right_gray)

            # Check for objects within the threshold distance
            self.detect_threshold_crossing(disparity_map, threshold_distance)

            if show_result:
                depth_map_visual = cv2.applyColorMap(cv2.convertScaleAbs(disparity_map, alpha=255.0 / np.max(disparity_map)), cv2.COLORMAP_JET)
                cv2.imshow("Depth Map", depth_map_visual)
                cv2.imshow("Left Color Image", img_left_color)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def detect_threshold_crossing(self, disparity_map, threshold_distance):
        """Detects objects within the threshold distance and triggers an alert."""
        for y in range(0, disparity_map.shape[0], 10):  # Sampling every 10 pixels vertically
            for x in range(0, disparity_map.shape[1], 10):  # Sampling every 10 pixels horizontally
                distance = self.calculate_distance(disparity_map, x, y)
                if distance is not None and distance < threshold_distance:
                    self.event_bus.publish("threshold_distance_crossed", distance=distance, object_name="object")

    def release(self):
        """Releases camera and display resources."""
        self.camera.close()
        cv2.destroyAllWindows()
