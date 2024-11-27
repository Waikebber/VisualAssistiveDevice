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
            raise ValueError(
                "Invalid method. Choose 'median', 'gaussian', or 'bilateral'."
            )

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
                object_distances.append(
                    (obj_name, distance, confidence, (center_x, center_y))
                )

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

                object_distances.append(
                    (obj_name, min_distance, confidence, (min_x, min_y))
                )

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
            return float("inf")  # Infinite distance if disparity is zero or invalid
            
    def solve_center_pixel_distance(self, disparity):
        """
        Use cv2.solve() to calculate the distance of the center pixel.

        Args:
            disparity (np.array): Disparity map with raw disparity values.
            focal_length (float): Focal length of the camera (in pixels).
            baseline (float): Baseline distance between the stereo cameras (in meters or cm).

        Returns:
            float: Distance of the center pixel (in the same unit as the baseline).
        """
        h, w = disparity.shape
        center_y, center_x = h // 2, w // 2  # Center pixel coordinates

        # Get the disparity value at the center pixel
        center_disparity = disparity[center_y, center_x]

        if center_disparity <= 0:
            return float('inf')  # Return infinity for invalid or zero disparity

        A = np.array([
            [self.focal_length, -center_x], 
            [self.focal_length, -(center_x - self.baseline * center_disparity / self.focal_length)]
        ], dtype=np.float64)

        b = np.array([0, 0], dtype=np.float64)

        # Solve the system of equations
        ret, sol = cv2.solve(A, b, flags=cv2.DECOMP_QR)

        # Extract Z (depth) from the solution
        _, Z = sol.flatten()
        return Z

    
    def detect_closest_object_from_colormap(self, disparity, disparity_color, min_region_area=4000):
        """
        Detect the closest object by isolating red shades from the disparity colormap and calculating the distance
        using the disparity map.
        
        Args:
            disparity (np.array): Grayscale disparity map with raw disparity values.
            disparity_color (np.array): Colormap of the disparity map (BGR format) used for visualization.
            min_region_area (int): Minimum area of a region to be considered a valid object.
        
        Returns:
            tuple: (closest_distance, (cX, cY)) where closest_distance is the distance to the closest object,
                   and (cX, cY) are the coordinates of the centroid of the closest object. If no valid object is found,
                   returns (float('inf'), None).
        """
        # Convert the disparity color image to HSV
        hsv_image = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2HSV)
        
        # Define the HSV range for detecting red color (typically red appears in two ranges)
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        # Create two masks to capture both red ranges
        mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        
        # Combine the masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Optional: Apply morphological operations to reduce noise in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of the detected closest regions
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        closest_distance = float('inf')
        closest_region_center = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_area :
                continue  # Ignore small regions

            # Calculate the centroid of the contour to represent the object's position
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Use a 5x5 region around the centroid to calculate the average disparity
                window_size = 10
                half_window = window_size // 2

                # Ensure the coordinates are within bounds
                start_x = max(0, cX - half_window)
                end_x = min(disparity.shape[1], cX + half_window + 1)
                start_y = max(0, cY - half_window)
                end_y = min(disparity.shape[0], cY + half_window + 1)

                # Extract the disparity values from the disparity map in the given window
                disparity_window = disparity[start_y:end_y, start_x:end_x]
                
                # Calculate the average disparity value
                valid_disparities = disparity_window[disparity_window > 0]  # Ignore zero or invalid disparity values
                if valid_disparities.size > 0:
                    avg_disparity = np.mean(valid_disparities)
                    min_disparity = np.min(valid_disparities)

                    # Calculate distance from disparity
                    distance = (self.focal_length * self.baseline) / min_disparity
                    print(f"Distance: {distance}m, BASELINE: {self.baseline}m, FOCAL: {self.focal_length}px, MIN_DISP: {min_disparity}")
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_region_center = (cX, cY)

        return closest_distance, closest_region_center

    def detect_closest_distance_disparity(self, disparity, min_region_area=4000):
        """
        Detect the closest object directly using the disparity map, based on regions with minimum disparity values.

        Args:
            disparity (np.array): Grayscale disparity map with raw disparity values.
            min_region_area (int): Minimum area of a region to be considered a valid object.

        Returns:
            tuple: (closest_distance, (cX, cY)) where closest_distance is the distance to the closest object,
                   and (cX, cY) are the coordinates of the centroid of the closest object. If no valid object is found,
                   returns (float('inf'), None).
        """
        # Threshold the disparity map to exclude invalid values (e.g., zero disparities)
        valid_disparity_mask = disparity > 0

        # Apply a binary threshold to create a mask of valid regions
        _, binary_mask = cv2.threshold(disparity, 1, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # Optional: Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of the detected regions
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        closest_distance = float('inf')
        closest_region_center = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_area :
                continue  # Ignore small regions

            # Calculate the centroid of the contour to represent the object's position
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Use a 5x5 region around the centroid to calculate the minimum disparity
                window_size = 10
                half_window = window_size // 2

                # Ensure the coordinates are within bounds
                start_x = max(0, cX - half_window)
                end_x = min(disparity.shape[1], cX + half_window + 1)
                start_y = max(0, cY - half_window)
                end_y = min(disparity.shape[0], cY + half_window + 1)

                # Extract the disparity values from the disparity map in the given window
                disparity_window = disparity[start_y:end_y, start_x:end_x]

                # Calculate the minimum valid disparity in the region
                valid_disparities = disparity_window[disparity_window > 0]
                if valid_disparities.size > 0:
                    min_disparity = np.min(valid_disparities)

                    # Calculate distance from disparity
                    distance = 1.5+ (self.focal_length * self.baseline) / min_disparity
                    print(f"Distance: {distance}m, BASELINE: {self.baseline}m, FOCAL: {self.focal_length}px, MIN_DISP: {min_disparity}")
                    if distance < closest_distance:
                        closest_distance = distance 
                        closest_region_center = (cX, cY)

        return closest_distance, closest_region_center
        
    def calculate_obstacle(self, depth_map, depth_thresh=100.0):
        """
        Detect obstacles based on the depth map and a safe distance threshold.

        Args:
            depth_map (np.array): The depth map with each pixel representing distance (in cm).
            depth_thresh (float): Threshold for the safe distance (in cm).

        Returns:
            dict: A dictionary containing:
                  - "warning": bool indicating if an obstacle is detected within the threshold.
                  - "bounding_box": Tuple (x, y, w, h) of the largest detected obstacle (or None).
                  - "avg_depth": Average depth of the obstacle (or None).
                  - "mask": The binary mask used for obstacle detection.
        """
        # Preprocess depth map to reduce noise
        depth_map = cv2.medianBlur(depth_map.astype(np.uint8), 5)

        # Create a binary mask for regions closer than the threshold
        mask = cv2.inRange(depth_map, 10, depth_thresh)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Check if any significant obstacle is detected
        if np.sum(mask) / 255.0 > 0.01 * mask.shape[0] * mask.shape[1]:
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(contours, key=cv2.contourArea, reverse=True)

            # Check the largest contour area
            max_area_threshold = 0.5 * mask.shape[0] * mask.shape[1]
            if cnts and cv2.contourArea(cnts[0]) > 0.01 * mask.shape[0] * mask.shape[1]:
                if cv2.contourArea(cnts[0]) > max_area_threshold:
                    return {"warning": False, "bounding_box": None, "avg_depth": None, "mask": mask}

                # Get bounding box of the largest contour
                x, y, w, h = cv2.boundingRect(cnts[0])

                # Create a mask for the largest contour
                mask2 = np.zeros_like(mask)
                cv2.drawContours(mask2, cnts, 0, (255), -1)

                # Calculate the average depth within the bounding box
                depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)

                return {
                    "warning": True,
                    "bounding_box": (x, y, w, h),
                    "avg_depth": depth_mean[0][0],
                    "mask": mask,
                }

        # No significant obstacle detected
        return {"warning": False, "bounding_box": None, "avg_depth": None, "mask": mask}
            
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
        disparity_grayscale = (disparity_map - local_min) * (
            65535.0 / (local_max - local_min)
        )
        disparity_fixtype = cv2.convertScaleAbs(
            disparity_grayscale, alpha=(255.0 / 65535.0)
        )
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

        # Draw bounding boxes and labels
        for obj_name, bbox, confidence in detected_objects:
            # Get box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in bbox[0]]

            # Draw bounding box
            cv2.rectangle(disparity_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label_size = cv2.getTextSize(obj_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                disparity_color,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                disparity_color,
                obj_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return disparity_color
        
