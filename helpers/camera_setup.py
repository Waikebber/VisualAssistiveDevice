from picamera import PiCamera
import numpy as np

def initialize_camera(cam_width, cam_height):
    """Initialize and configure the stereo camera."""
    camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
    camera.resolution = (cam_width, cam_height)
    camera.framerate = 20
    camera.hflip = True
    return camera

def setup_capture_buffer(img_height, img_width):
    """Create capture buffer for camera."""
    return np.zeros((img_height, img_width, 4), dtype=np.uint8)