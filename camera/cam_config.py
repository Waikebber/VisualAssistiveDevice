from picamera2 import Picamera2

def get_camera_config(camera_id: int, camera: Picamera2, width: int, height: int):
    """
    Get camera configuration for stereo vision setup.
    
    Args:
        camera_id (int): Camera identifier (0 for left, 1 for right)
        camera (Picamera2): Picamera2 instance to configure
        width (int): Desired frame width
        height (int): Desired frame height
        
    Returns:
        dict: Camera configuration dictionary
    """
    # Ensure dimensions are valid (divisible by 32 for width, 16 for height)
    width = int((width + 31) / 32) * 32
    height = int((height + 15) / 16) * 16
    
    config = camera.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
        raw={"size": (1640, 1232), "format": "SBGGR10_CSI2P"},
        buffer_count=4,
        controls={
            "FrameDurationLimits": (23894, 11767556),
            "NoiseReductionMode": 0,
            "AwbEnable": True,
            "AeEnable": True,
            "AnalogueGain": 1.0,
            "FrameRate": 20.0
        }
    )
    
    print(f"Camera {camera_id} configuration:")
    print(f"- Resolution: {width}x{height}")
    print(f"- Raw format: {config['raw']['format']}")
    print(f"- Main format: {config['main']['format']}")
    
    return config

def initialize_camera(camera_id: int, width: int, height: int) -> Picamera2:
    """
    Initialize and configure a single camera.
    
    Args:
        camera_id (int): Camera identifier (0 for left, 1 for right)
        width (int): Desired frame width
        height (int): Desired frame height
        
    Returns:
        Picamera2: Configured camera instance
    """
    camera = Picamera2(camera_id)
    config = get_camera_config(camera_id, camera, width, height)
    camera.configure(config)
    return camera
