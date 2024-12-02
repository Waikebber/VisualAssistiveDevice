import os, json, multiprocessing
import cv2
import numpy as np
import RPi.GPIO as GPIO
from speakers import speak
from stereo_calibration.camera.cam_config import initialize_camera
from image_rec.img_rec import ImgRec
from stereo_calibration.rectify import get_closest_distance, make_colored_distance_map, rectify_imgs, make_disparity_map
from image_rec.stereoImgRec import create_detection_image, calculate_object_distances
from scipy.ndimage import median_filter

CONFIDENCE = 0.6
THRESHOLD = 2.5   # Threshold in meters (2.5m)
CONFIG_FILE = "stereo_calibration/cam_config.json"
CALIB_RESULTS = 'data/stereo_images/scenes/calibration_results'
SAVE_OUTPUT = True
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'output.png'
DISPLAY_RATIO = 1  # Scaling factor for display
BORDER = 50        # Border to ignore for depth map calculations

# Load configuration from config.json
config_path = CONFIG_FILE
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
BASELINE = int(config['baseline_length_mm']) / 1000  # Baseline in m (60mm)
FOCAL = int(config['focal_length_mm']) / 1000        # Focal length in m (2.6mm)
SENSOR_WIDTH = float(config['cmos_size_m'])          # Sensor width in m (1/4in)
H_FOV = int(config['field_of_view']['horizontal'])   # Horizontal field of view (73deg)
scale_ratio = float(config['scale_ratio'])           # Image scaling ratio (0.5)
cam_width = int(config['image_width'])
cam_height = int(config['image_height'])

DISTANCE_SCALE = BASELINE * 5 / 8

# Camera resolution / image scaling / focal length
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Used camera resolution: " + str(cam_width) + " x " + str(cam_height))
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Image resolution: " + str(img_width) + " x " + str(img_height))

# Initialize the image recognition model and distance calculator
img_recognizer = ImgRec()

# GPIO Setup
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 26
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button_press_action():
    global current_distances, rectified_pair
    distances_on_button_press = current_distances
    current_pair = rectified_pair
    left_rectified = current_pair[0]
    
    if current_pair is None or distances_on_button_press is None:
        print("No frames available yet")
        return
        
    print("Button pressed - performing object detection and distance measurement")
    
    detected_objects = img_recognizer.predict_frame(left_rectified)
    print('OBJECTS')
    print(detected_objects)

    if SAVE_OUTPUT:
        output = create_detection_image(distances_on_button_press, detected_objects, BORDER)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FILE), output)
    
    # Calculate distances for detected objects using current distance map
    object_distances = calculate_object_distances(distances_on_button_press, detected_objects, border=0, percentile=20, confidence_threshold=CONFIDENCE)
    
    # Process and announce detected objects within threshold
    for obj_name, distance_val in object_distances:
        if distance_val < THRESHOLD:
            message = f"Detected {obj_name} at {distance_val:.2f} meters"
            print(message)
            speak_async(message)
        else:
            message = f"{obj_name} further than {THRESHOLD} meters"
            print(message)
            speak_async(message)

def on_button_press(channel):
    p = multiprocessing.Process(target=button_press_action)
    p.start()

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=on_button_press, bouncetime=200)

# Initialize the cameras
camera_left = initialize_camera(1, img_width, img_height)
camera_right = initialize_camera(0, img_width, img_height)

# Start cameras
camera_left.start()
camera_right.start()

# Initialize interface windows
cv2.namedWindow("Depth Map")
cv2.moveWindow("Depth Map", 50, 100)
cv2.namedWindow("Left Camera")
cv2.moveWindow("Left Camera", 450, 100)
cv2.namedWindow("Right Camera")
cv2.moveWindow("Right Camera", 850, 100)

audio_process = None
def speak_async(text):
    global audio_process
    if audio_process is None or not audio_process.is_alive():
        audio_process = multiprocessing.Process(target=speak, args=(text, 3, 90))
        audio_process.start()

try:
    # Start the main processing loop
    while True:
        # Capture frames
        current_frame_left = camera_left.capture_array()
        current_frame_right = camera_right.capture_array()
    
        # Rectify the stereo pair
        left_rectified, right_rectified, Q, focal_length = rectify_imgs(current_frame_left, current_frame_right, CALIB_RESULTS)

        # Generate the disparity map
        min_disp = 0
        num_disp = 16 * 2  # must be divisible by 16
        block_size = 10
        disparity = make_disparity_map(left_rectified, right_rectified, min_disp, num_disp, block_size)
        rectified_pair = (left_rectified, right_rectified)
        
        # Convert disparity to depth map
        depth_map = cv2.reprojectImageTo3D(disparity, Q)
        current_distances = depth_map[:, :, 2] * DISTANCE_SCALE

        closest_distance, closest_coordinates = get_closest_distance(current_distances)
        
        if closest_distance is not None and closest_coordinates is not None and closest_distance < THRESHOLD:
            print(f'Closest distance: {closest_distance:.2f} meters at coordinates {closest_coordinates}')
            speak_async(f'Closest distance: {closest_distance:.2f} meters')
        
        # Create a colored depth map for visualization
        depth_map_colored = make_colored_distance_map(current_distances, min_distance=0, max_distance=THRESHOLD)
        
        # Draw the closest point on the left rectified image and disparity map
        if closest_coordinates is not None:
            cv2.circle(left_rectified, closest_coordinates, 5, (0, 255, 0), 2)
            cv2.circle(current_distances, closest_coordinates, 5, (0, 255, 0), 2)

        # Display frames
        cv2.imshow("Left Camera", left_rectified)
        cv2.imshow("Right Camera", right_rectified)
        cv2.imshow("Depth Map", depth_map_colored)

        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except Exception as e:
    # Log any errors that occur during the loop
    print(f"Error during processing: {e}")

finally:
    # Ensure cleanup happens no matter what
    camera_left.stop()
    camera_right.stop()
    cv2.destroyAllWindows()
