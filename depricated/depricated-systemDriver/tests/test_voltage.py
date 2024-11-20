import RPi.GPIO as GPIO
import time

# Set up GPIO using BCM numbering
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin you want to use
OUTPUT_PIN = 12  # GPIO 26 (pin 32 on the board)

# Set up the pin as an output
GPIO.setup(OUTPUT_PIN, GPIO.OUT)

try:
    while True:
        # Turn on the GPIO pin (set to HIGH)
        GPIO.output(OUTPUT_PIN, GPIO.HIGH)
        print("GPIO pin {} is ON".format(OUTPUT_PIN))
        time.sleep(5)  # Wait for 2 seconds

        # Turn off the GPIO pin (set to LOW)
        GPIO.output(OUTPUT_PIN, GPIO.LOW)
        print("GPIO pin {} is OFF".format(OUTPUT_PIN))
        time.sleep(5)  # Wait for 2 seconds

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    # Clean up GPIO on program exit
    GPIO.cleanup()
    print("GPIO cleanup completed")
