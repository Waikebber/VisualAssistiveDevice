import json
import os

def load_camera_config(config_path="cam_config.json"):
    """Load and validate camera configuration from JSON file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Calculate derived parameters
    params = {
        'BASELINE': int(config['baseline_length_mm']) / 1000,
        'FOCAL': int(config['focal_length_mm']) / 1000,
        'SENSOR_WIDTH': float(config['cmos_size_m']),
        'H_FOV': int(config['field_of_view']['horizontal']),
        'scale_ratio': float(config['scale_ratio']),
        'cam_width': int((int(config['image_width']) + 31) / 32) * 32,
        'cam_height': int((int(config['image_height']) + 15) / 16) * 16,
        'THRESHOLD': 2.5  # Threshold in meters
    }
    
    # Calculate scaled dimensions
    params['img_width'] = int(params['cam_width'] * params['scale_ratio'])
    params['img_height'] = int(params['cam_height'] * params['scale_ratio'])
    
    return params