import numpy as np
import cv2
from scipy.stats import norm

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
        
    def reduce_noise(self, disparity_map, method="median", kernel_size=5):
        """Reduce noise in the disparity map using filtering.
        
        Args:
            disparity_map (np.array): Input disparity map
            method (str): Noise reduction method ('median', 'gaussian', or 'bilateral')
            kernel_size (int): Kernel size for the filter
            
        Returns:
            np.array: Denoised disparity map
        """
        if method == "median":
            return cv2.medianBlur(disparity_map, kernel_size)
        elif method == "gaussian":
            return cv2.GaussianBlur(disparity_map, (kernel_size, kernel_size), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(disparity_map, kernel_size, 75, 75)
        else:
            raise ValueError("Invalid method. Choose 'median', 'gaussian', or 'bilateral'.")


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
            normalized_disparity = disparity_map[center_y, center_x] + 60.0
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
            normalized_disparity = box_disparity + 60.0
            
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
    
    def calculate_center_distance(self, disparity_map):
        """Calculate distance for the center pixel.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            
        Returns:
            float: Distance to center pixel in meters, or inf if invalid
        """
        if disparity_map is None or disparity_map.size == 0:
            raise ValueError("Invalid or empty disparity map provided.")

        # Get the center pixel coordinates
        center_x = disparity_map.shape[1] // 2
        center_y = disparity_map.shape[0] // 2

        # Fetch the disparity value for the center pixel
        center_disparity = disparity_map[center_y, center_x]

        # Apply normalization to handle negative values or offsets
        normalized_disparity = center_disparity + 60.0

        if normalized_disparity > 0:  # Ensure disparity is positive
            # Calculate the distance using focal length, baseline, and disparity
            distance = (self.focal_length * self.baseline) / normalized_disparity
            return distance
        else:
            return float('inf')  # Infinite distance if disparity is zero or invalid
    
    def calculate_distance(self, disparity_map):
        """Calculate the closest distance in the map.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            
        Returns:
            tuple: (min_distance, (min_x, min_y)) containing the minimum distance
                  and its coordinates, or (inf, None) if no valid distances
        """
        normalized_disparity = disparity_map + 60.0
        valid_disparities = normalized_disparity[normalized_disparity > 0]
        
        if len(valid_disparities) > 0:
            min_disparity = valid_disparities.min()
            min_distance = (self.focal_length * self.baseline) / min_disparity
            
            # Find coordinates of minimum distance
            min_idx = np.where(normalized_disparity == min_disparity)
            min_y, min_x = min_idx[0][0], min_idx[1][0]
            
            return min_distance, (min_x, min_y) + 1
        else:
            return float('inf'), None

    def analyze_disparity_distribution(self, disparity_map):
        """Analyze the disparity map to find the closest object using Gaussian distribution.
        
        Args:
            disparity_map (np.array): Disparity map from stereo vision
            
        Returns:
            tuple: (closest_distance, mean_distance, std_distance)
                   Closest distance is the value in the left tail of the distribution.
        """
        # Normalize the disparity map
        normalized_disparity = disparity_map + 61.0
        valid_disparities = normalized_disparity[normalized_disparity > 0]
        
        if valid_disparities.size == 0:
            return float('inf'), None, None  # No valid disparities
        
        # Convert disparities to distances
        distances = (self.focal_length * self.baseline) / valid_disparities
        
        # Fit Gaussian distribution to distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Identify the closest distance (e.g., mean - 2 * std_dev)
        cutoff_distance = mean_distance - 2 * std_distance
        closest_distance = distances[distances <= cutoff_distance].min(initial=float('inf'))
        
        return closest_distance, mean_distance, std_distance
    
    def find_closest_valid_region(self, disparity_map, min_region_size=500):
        """Find the closest valid region in the disparity map.

        Args:
            disparity_map (np.array): Disparity map from stereo vision
            min_region_size (int): Minimum size of the region to be considered valid

        Returns:
            tuple: (closest_distance, (x, y), region_size) or (inf, None, 0)
        """
        # Normalize the disparity map and convert to 8-bit
        normalized_disparity = cv2.normalize(disparity_map, None, alpha=0, beta=255, 
                                             norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold the disparity map to get regions with valid disparities
        _, thresh = cv2.threshold(normalized_disparity, 0, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)

        closest_distance = float('inf')
        closest_region_center = None
        largest_region_size = 0

        for label in range(1, num_labels):  # Skip background label 0
            region_size = stats[label, cv2.CC_STAT_AREA]
            if region_size < min_region_size:
                continue  # Ignore small regions

            # Get the average disparity of the region
            mask = (labels == label)
            region_disparities = disparity_map[mask]

            # Calculate the mean disparity and corresponding distance
            mean_disparity = np.mean(region_disparities)
            if mean_disparity <= 0:
                continue

            distance = (self.focal_length * self.baseline) / (mean_disparity + 1e-6)

            if distance < closest_distance:
                closest_distance = distance
                closest_region_center = centroids[label]
                largest_region_size = region_size
        if visualize:
            # Create a color map to display regions
            region_display = cv2.cvtColor(normalized_disparity, cv2.COLOR_GRAY2BGR)
    
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_region_size:
                    mask = (labels == label)
                    region_display[mask] = [0, 255, 0]  # Color valid regions green
    
            cv2.imshow("Regions", region_display)

        return closest_distance, closest_region_center, largest_region_size
    
    
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
