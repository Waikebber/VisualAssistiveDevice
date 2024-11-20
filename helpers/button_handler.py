import RPi.GPIO as GPIO
import multiprocessing

class ButtonHandler:
    def __init__(self, pin=21):
        self.BUTTON_PIN = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.setup_button_handler()
    
    def button_press_action(self):
        print("button pressed subprocess started")
    
    def on_button_press(self, channel):
        p = multiprocessing.Process(target=self.button_press_action)
        p.start()
    
    def setup_button_handler(self):
        GPIO.add_event_detect(
            self.BUTTON_PIN, 
            GPIO.RISING, 
            callback=self.on_button_press, 
            bouncetime=200
        )
    
    def cleanup(self):
        GPIO.cleanup()

class ButtonHandlerPlaceholder(ButtonHandler):
    """Placeholder button handler for testing without image recognition"""
    def button_press_action(self):
        print("Button pressed - placeholder handler")

class ImageRecButtonHandler(ButtonHandler):
    def __init__(self, stereo_processor, img_rec, baseline, confidence=0.5):
        super().__init__()
        self.stereo_processor = stereo_processor
        self.img_rec = img_rec
        self.current_frame = None
        self.current_disparity = None
        self.confidence = confidence
        self.baseline = baseline
        
    def button_press_action(self):
        if self.current_frame is not None and self.current_disparity is not None:
            print("Processing image recognition...")
            # Perform image recognition
            detected_objects = self.img_rec.predict_frame(self.current_frame, self.confidence)
            
            if not detected_objects:
                print("No objects detected")
                return
                
            # Calculate distances for each detected object
            for obj_name, bbox, confidence in detected_objects:
                try:
                    # Calculate object distance using bounding box
                    distance = self.stereo_processor.calculate_object_distance(
                        self.current_disparity,
                        bbox[0].numpy()[0],  # Convert bbox tensor to numpy array
                        self.baseline
                    )
                    
                    # Format and print results
                    dist_m = round(distance, 3)
                    dist_ft = round(distance * 3.281, 3)
                    print(f"Detected {obj_name} (conf: {confidence:.2f}) at {dist_m}m = {dist_ft}ft")
                except Exception as e:
                    print(f"Error calculating distance for {obj_name}: {str(e)}")