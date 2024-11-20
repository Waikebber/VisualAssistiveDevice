import RPi.GPIO as GPIO
import time

# Set up GPIO
pwm_pin = 12 # GPIO 26 corresponds to physical pin 32
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup(pwm_pin, GPIO.OUT)  # Set the pin as an output

# Set PWM frequency to 700 Hz
freq = 700
pwm = GPIO.PWM(pwm_pin, freq)  # Create a PWM instance with the specified frequency
pwm.start(50)  # Start PWM with a 50% duty cycle
print(f"Started! w Pin {pwm_pin} & freq: {freq}")

try:
    while True:
        time.sleep(1) # Run indefinitely
except KeyboardInterrupt:
    pass
finally:
    pwm.stop() # Stop PWM
    GPIO.cleanup()
