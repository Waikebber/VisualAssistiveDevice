import cv2
import numpy as np
import json
from math import tan, pi

class StereoProcessor:
    def __init__(self, img_width, img_height, h_fov, baseline):
        self.img_width = img_width
        self.img_height = img_height
        self.h_fov = h_fov
        self.baseline = baseline
        self.focal_length_px = (img_width * 0.5) / tan(h_fov * 0.5 * pi / 180)
        self.sbm = cv2.StereoBM_create()
        self.load_default_params()
    
    def load_default_params(self):
        """Initialize default stereo matching parameters."""
        self.sbm.setPreFilterType(1)
        self.sbm.setPreFilterSize(9)  # PFS
        self.sbm.setPreFilterCap(29)  # PFC
        self.sbm.setMinDisparity(-30) # MDS
        self.sbm.setNumDisparities(16 * 9)  # NOD
        self.sbm.setTextureThreshold(100)  # TTH
        self.sbm.setUniquenessRatio(10)    # UR
        self.sbm.setSpeckleRange(14)       # SR
        self.sbm.setSpeckleWindowSize(100) # SPWS
        self.sbm.setBlockSize(15)          # SWS
    
    def load_params_from_file(self, filename):
        """Load stereo matching parameters from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.sbm.setBlockSize(data['SADWindowSize'])
            self.sbm.setPreFilterSize(data['preFilterSize'])
            self.sbm.setPreFilterCap(data['preFilterCap'])
            self.sbm.setMinDisparity(data['minDisparity'])
            self.sbm.setNumDisparities(data['numberOfDisparities'])
            self.sbm.setTextureThreshold(data['textureThreshold'])
            self.sbm.setUniquenessRatio(data['uniquenessRatio'])
            self.sbm.setSpeckleRange(data['speckleRange'])
            self.sbm.setSpeckleWindowSize(data['speckleWindowSize'])
    
    def calculate_center_distance(self, disparity):
        """Calculate distance for center pixel."""
        normalized_disparity = disparity + 61.0
        center_x = normalized_disparity.shape[1] // 2
        center_y = normalized_disparity.shape[0] // 2
        center_disparity = normalized_disparity[center_y, center_x]
        
        if center_disparity > 0:
            return (self.focal_length_px * self.baseline) / center_disparity
        return float('inf')
    
    def calculate_object_distance(self, disparity, bbox):
        """Calculate distance for an object given its bounding box.
        
        Args:
            disparity (np.array): Raw disparity map
            bbox (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            float: Distance to the object in meters
        """
        normalized_disparity = disparity + 61.0
        
        # Calculate center point of bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        # Ensure coordinates are within image bounds
        center_x = min(max(center_x, 0), normalized_disparity.shape[1] - 1)
        center_y = min(max(center_y, 0), normalized_disparity.shape[0] - 1)
        
        # Get disparity value at object center
        object_disparity = normalized_disparity[center_y, center_x]
        
        if object_disparity > 0:
            return (self.focal_length_px * self.baseline) / object_disparity
        return float('inf')
    
    def compute_depth_map(self, rectified_pair, return_raw_disparity=False):
        """Compute and visualize depth map."""
        dmLeft, dmRight = rectified_pair
        disparity = self.sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
        
        # Normalize disparity map for visualization
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_grayscale = (disparity - local_min) * (65535.0 / (local_max - local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        
        if return_raw_disparity:
            return disparity_color, disparity
        
        # Calculate center distance
        center_distance = self.calculate_center_distance(disparity)
        return disparity_color, center_distance