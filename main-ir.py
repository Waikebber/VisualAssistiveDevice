import cv2
from stereovision.calibration import StereoCalibration
from helpers.config import load_camera_config
from helpers.stereo_processing import StereoProcessor
from helpers.camera_setup import initialize_camera, setup_capture_buffer
from helpers.gui import initialize_windows, update_displays
from helpers.button_handler import ImageRecButtonHandler
from helpers.audio import AudioHandler
from image_rec.img_rec import ImgRec

THRESHOLD = 1.0  # meters 
CONFIG_FILE = "stereo-calibration/cam_config.json"
SETTINGS_FILE = "stereo-calibration/3dmap_set.txt"
CALIB_RESULTS = 'stereo-calibration/calib_result'
DISPLAY = True
CONFIDENCE = 0.6

def main():
    try:
        # Load configuration
        params = load_camera_config(CONFIG_FILE)
        params['THRESHOLD'] = THRESHOLD
        
        # Initialize components
        stereo_processor = StereoProcessor(
            params['img_width'],
            params['img_height'],
            params['H_FOV'],
            params['BASELINE']
        )
        
        # Initialize image recognition
        img_rec = ImgRec()
        
        # Initialize button handler with stereo processor and image recognition
        button_handler = ImageRecButtonHandler(
            stereo_processor, 
            img_rec, 
            params['BASELINE'],
            CONFIDENCE
        )
        audio_handler = AudioHandler()
        
        # Load stereo calibration
        print('Read calibration data and rectifying stereo pair...')
        calibration = StereoCalibration(input_folder=CALIB_RESULTS)
        
        # Initialize camera and capture buffer
        camera = initialize_camera(params['cam_width'], params['cam_height'])
        capture = setup_capture_buffer(params['img_height'], params['img_width'])
        
        # Initialize display windows if needed
        if DISPLAY:
            initialize_windows()
        
        # Load stereo matching parameters
        stereo_processor.load_params_from_file(SETTINGS_FILE)
        
        print("System initialized. Press button to perform object detection.")
        
        # Main processing loop
        for frame in camera.capture_continuous(
            capture,
            format="bgra",
            use_video_port=True,
            resize=(params['img_width'], params['img_height'])
        ):
            # Convert and split stereo pair
            pair_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgLeft = pair_img[0:params['img_height'], 0:int(params['img_width'] / 2)]
            imgRight = pair_img[0:params['img_height'], int(params['img_width'] / 2):params['img_width']]
            
            # Store the left image for image recognition
            button_handler.current_frame = imgLeft.copy()
            
            # Process stereo images
            rectified_pair = calibration.rectify((imgLeft, imgRight))
            disparity, raw_disparity = stereo_processor.compute_depth_map(
                rectified_pair,
                return_raw_disparity=True
            )
            
            # Store the disparity map for distance calculations
            button_handler.current_disparity = raw_disparity.copy()
            
            # Calculate center distance for proximity warning
            center_distance = stereo_processor.calculate_center_distance(raw_disparity)
            
            # Check distance threshold
            if center_distance < params['THRESHOLD']:
                thresh_ft = round(params['THRESHOLD'] * 3.281, 3)
                dist_ft = round(center_distance * 3.281, 3)
                message = f"Threshold({params['THRESHOLD']}m={thresh_ft}ft) breached, center: {center_distance}m = {dist_ft}ft"
                print(message)
                audio_handler.speak_async(message)
            
            # Update displays if enabled
            if DISPLAY:
                key = update_displays(disparity, imgLeft, imgRight)
                if key == ord("q"):
                    break
    
    finally:
        # Cleanup
        if DISPLAY:
            cv2.destroyAllWindows()
        button_handler.cleanup()

if __name__ == "__main__":
    main()