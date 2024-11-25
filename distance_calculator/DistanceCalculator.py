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
    
    def detect_closest_object(
        self, disparity_map, disparity_threshold, min_region_area=500
    ):
        # Create a binary mask where disparity is greater than the threshold
        close_objects_mask = cv2.threshold(
            disparity_map, disparity_threshold, 255, cv2.THRESH_BINARY
        )[1].astype(np.uint8)

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close_objects_mask = cv2.morphologyEx(
            close_objects_mask, cv2.MORPH_OPEN, kernel
        )

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            close_objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        closest_distance = float("inf")
        closest_region_center = None
        largest_region_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_region_area:
                continue  # Ignore small regions

            # Calculate the mask for the current contour
            contour_mask = np.zeros_like(disparity_map, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, color=1, thickness=-1)

            # Calculate the average disparity within the contour
            mean_disparity = cv2.mean(disparity_map, mask=contour_mask)[0]

            if mean_disparity <= 0:
                continue

            # Calculate distance
            distance = (self.focal_length * self.baseline) / mean_disparity

            if distance < closest_distance:
                closest_distance = distance
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    closest_region_center = (cX, cY)
                else:
                    closest_region_center = None
                largest_region_area = area

        return closest_distance, closest_region_center, largest_region_area
        
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
