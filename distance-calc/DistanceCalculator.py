import numpy as np
import cv2

"""File to calculate distances from disparity map.
"""

class DistanceCalculator:
    def __init__(self, baseline, focal_length):
        """Initialize the DistanceCalculator with baseline and focal length.
        
        Args:
            baseline (float): Baseline distance between cameras in meters
            focal_length (float): Focal length in pixels
        """
        self.baseline = baseline
        self.focal_length = focal_length

    def calculate_object_center_distances(self, disparity_map, detected_objects):
        """Calculate center distances of bounding boxes from detected objects.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            detected_objects (list): List of tuples containing (obj_name, bbox, confidence)
            
        Returns:
            list: List of tuples containing (obj_name, distance, confidence, (center_x, center_y))
        """
        object_distances = []
        height, width = disparity_map.shape
        
        for obj_name, bbox, confidence in detected_objects:
            # Get center coordinates of the bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox[0]]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Ensure coordinates are within bounds
            center_x = min(max(center_x, 0), width - 1)
            center_y = min(max(center_y, 0), height - 1)
            
            # Calculate distance for object center point
            normalized_disparity = disparity_map[center_y, center_x] + 61.0
            if normalized_disparity > 0:
                distance = (self.focal_length * self.baseline) / normalized_disparity
                object_distances.append((obj_name, distance, confidence, (center_x, center_y)))
        
        return object_distances
    
    def calculate_object_distances(self, disparity_map, detected_objects):
        """Calculate minimum distances for detected objects using the disparity map.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            detected_objects (list): List of tuples containing (obj_name, bbox, confidence)
            
        Returns:
            list: List of tuples containing (obj_name, min_distance, confidence, (min_x, min_y))
        """
        object_distances = []
        height, width = disparity_map.shape
        
        for obj_name, bbox, confidence in detected_objects:
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in bbox[0]]
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Get disparity values within the bounding box
            box_disparity = disparity_map[y1:y2, x1:x2]
            normalized_disparity = box_disparity + 61.0
            
            # Find the minimum valid distance in the box
            valid_disparities = normalized_disparity[normalized_disparity > 0]
            if len(valid_disparities) > 0:
                min_disparity = valid_disparities.min()
                min_distance = (self.focal_length * self.baseline) / min_disparity
                
                # Find coordinates of minimum distance point
                min_idx = np.where(normalized_disparity == min_disparity)
                min_y, min_x = min_idx[0][0] + y1, min_idx[1][0] + x1
                
                object_distances.append((obj_name, min_distance, confidence, (min_x, min_y)))
        
        return object_distances
    
    def calculate_center_distance(self, disparity):
        """Calculate distance for the center pixel.
        
        Args:
            disparity (np.array): Disparity map from stereo vision
            
        Returns:
            float: Distance to center pixel in meters, or inf if invalid
        """
        # Get the center pixel coordinates
        normalized_disparity = disparity + 61.0
        center_x = normalized_disparity.shape[1] // 2
        center_y = normalized_disparity.shape[0] // 2
        center_disparity = normalized_disparity[center_y, center_x]
        
        if center_disparity > 0:  # Ensure disparity is positive
            # Calculate the distance Z
            distance = (self.focal_length * self.baseline) / center_disparity
            return distance
        else:
            return float('inf')  # Infinite distance if disparity is zero or negative
    
    def calculate_distance(self, disparity_map):
        """Calculate the closest distance in the map.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            
        Returns:
            tuple: (min_distance, (min_x, min_y)) containing the minimum distance
                  and its coordinates, or (inf, None) if no valid distances
        """
        normalized_disparity = disparity_map + 61.0
        valid_disparities = normalized_disparity[normalized_disparity > 0]
        
        if len(valid_disparities) > 0:
            min_disparity = valid_disparities.min()
            min_distance = (self.focal_length * self.baseline) / min_disparity
            
            # Find coordinates of minimum distance
            min_idx = np.where(normalized_disparity == min_disparity)
            min_y, min_x = min_idx[0][0], min_idx[1][0]
            
            return min_distance, (min_x, min_y)
        else:
            return float('inf'), None
    
    def create_detection_image(self, disparity_map, detected_objects):
        """Create visualization of disparity map with detected objects.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            detected_objects (list): List of tuples containing (obj_name, bbox, confidence)
            
        Returns:
            np.array: Color image with bounding boxes and labels
        """
        # Normalize and colorize disparity map
        local_max = disparity_map.max()
        local_min = disparity_map.min()
        disparity_grayscale = (disparity_map - local_min) * (65535.0 / (local_max - local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        
        # Draw bounding boxes and labels
        for obj_name, bbox, confidence in detected_objects:
            # Get box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in bbox[0]]
            
            # Draw bounding box
            cv2.rectangle(disparity_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label_size = cv2.getTextSize(obj_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(disparity_color, 
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1)
            
            # Draw label text
            cv2.putText(disparity_color,
                    obj_name,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2)
        
        return disparity_color