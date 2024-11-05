import threading
import time
from core.event_bus import EventBus

try:
    import RPi.GPIO as GPIO
except ImportError:
    # Mock GPIO for testing on non-Raspberry Pi platforms
    class GPIO:
        BOARD = IN = PUD_UP = FALLING = None
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode, pull_up_down=None): pass
        @staticmethod
        def add_event_detect(pin, edge, callback=None, bouncetime=0): pass
        @staticmethod
        def cleanup(): pass

class ButtonHandler:
    def __init__(self, event_bus: EventBus, button_pin=18, debounce_time=300, use_keypress=False):
        """Initialize button handler with GPIO pin, debounce time, and optional keypress simulation."""
        self.event_bus = event_bus
        self.button_pin = button_pin
        self.debounce_time = debounce_time
        self.use_keypress = use_keypress
        self._setup_gpio()

        if self.use_keypress:
            # Start a separate thread to listen for keypress
            self.keypress_thread = threading.Thread(target=self._listen_for_keypress)
            self.keypress_thread.daemon = True
            self.keypress_thread.start()

    def _setup_gpio(self):
        """Set up GPIO for button input, if available."""
        if GPIO != None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(self.button_pin, GPIO.FALLING, callback=self._button_callback, bouncetime=self.debounce_time)
            print(f"Button handler set up on GPIO pin {self.button_pin}")

    def _button_callback(self, channel):
        """Callback function triggered by a physical button press."""
        print("Button pressed!")
        self.event_bus.publish("button_pressed")

    def _listen_for_keypress(self):
        """Listen for keypress 'r' to simulate button press."""
        print("Listening for 'r' keypress to simulate button press.")
        while True:
            key = input("Press 'r' to simulate button press: ").strip().lower()
            if key == 'r':
                print("Simulated button press!")
                self.event_bus.publish("button_pressed")

    def cleanup(self):
        """Clean up GPIO resources."""
        GPIO.cleanup()
        print("Button handler cleaned up")
