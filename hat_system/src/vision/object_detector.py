from .glasses import ImgRec
import numpy as np
from core.event_bus import EventBus
from vision.stereo_processor import StereoProcessor

class ObjectDetector:
    def __init__(self, event_bus: EventBus, stereo_processor: StereoProcessor):
        self.event_bus = event_bus
        self.stereo_processor = stereo_processor
        self.img_rec = ImgRec()  # Initialize image recognition model

        # Subscribe to button press events to trigger detection
        self.event_bus.subscribe("button_pressed", self.on_button_press)

    def on_button_press(self):
        """Handles the button press event by capturing a frame, running image recognition, 
        and providing audio feedback with detected objects and distances."""
        
        # Capture the current color and grayscale frames from the stereo processor
        (img_left_color, _), (img_left_gray, img_right_gray) = self.stereo_processor.get_current_frame()

        # Run image recognition on the left color image
        detected_objects = self.img_rec.predict_frame(img_left_color, confidence_threshold=0.5)

        # Calculate distances using the grayscale images
        disparity_map = self.stereo_processor.calculate_disparity(img_left_gray, img_right_gray)
        results_with_distance = []
        for obj_name, bounding_box, confidence in detected_objects:
            # Calculate distance for the bounding box center
            x_min, y_min, x_max, y_max = map(int, bounding_box)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            distance = self.stereo_processor.calculate_distance(disparity_map=disparity_map, x=center_x, y=center_y)
            
            results_with_distance.append((obj_name, distance, confidence))
        
        # Publish the recognition results to the EventBus
        self.event_bus.publish("object_recognition_completed", results_with_distance=results_with_distance)
