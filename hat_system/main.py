import threading
import time
from core import EventBus
from audio.sound_manager import SoundManager
from vision.stereo_processor import StereoProcessor
from vision.object_detector import ObjectDetector
from io.button_handler import ButtonHandler

def main():
    event_bus = EventBus()

    sound_manager = SoundManager(event_bus)
    stereo_processor = StereoProcessor(event_bus=event_bus)
    object_detector = ObjectDetector(event_bus, stereo_processor)
    button_handler = ButtonHandler(event_bus, use_keypress=True)  # Set to `True` for keypress simulation, `False` for GPIO

    # Start continuous stereovision processing in a separate thread
    def run_stereo():
        stereo_processor.run_stereo_processing(threshold_distance=1.0, show_result=False)
    
    stereo_thread = threading.Thread(target=run_stereo)
    stereo_thread.daemon = True
    stereo_thread.start()

    # Keep the main thread alive to listen for events and manage cleanup
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        button_handler.cleanup()
        stereo_processor.release()

if __name__ == "__main__":
    main()
