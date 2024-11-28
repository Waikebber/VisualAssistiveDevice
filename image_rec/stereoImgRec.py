import cv2
import numpy as np

def create_detection_image(output_image, detected_objects, border=0):
    """
    Draw bounding boxes and labels on the detected objects in the provided image,
    considering the specified border to keep boxes within valid bounds.

    Parameters:
    output_image (numpy.ndarray): The image on which to draw the detected objects.
    detected_objects (list of tuples): A list of detected objects, where each tuple contains:
        - obj_name (str): The name of the detected object.
        - box (tuple): The bounding box of the object, in the format (x, y, w, h).
        - confidence (float): The confidence score of the detection.
    border (int, optional): The number of pixels to ignore from each edge of the image (default is 0).

    Returns:
    numpy.ndarray: The output image with bounding boxes and labels drawn for each detected object.
    """
    # Get the dimensions of the image
    height, width = output_image.shape[:2]

    # Loop through detected objects
    for obj in detected_objects:
        obj_name, box, confidence = obj
        x, y, w, h = map(int, box[0])

        # Adjust the bounding box coordinates to stay within the image bounds, considering the border
        x = max(border, x)
        y = max(border, y)
        x2 = min(width - border, x + w)
        y2 = min(height - border, y + h)

        # Draw bounding box around the detected object, only if it is within the valid area
        if x < x2 and y < y2:
            cv2.rectangle(output_image, (x, y), (x2, y2), (255, 0, 0), 2)
            # Put label with the object's name and confidence score
            label_position = (x, max(y - 10, border))  # Make sure the label is within the border
            cv2.putText(output_image, f"{obj_name}: {confidence:.2f}", label_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return output_image

def calculate_object_distances(distances, detected_objects, border=0, percentile=20, confidence_threshold=0.6):
    """
    Calculate the estimated distances to detected objects using a depth map.

    Parameters:
    distances (numpy.ndarray): A depth map containing distance values for each pixel.
    detected_objects (list of tuples): A list of detected objects, where each tuple contains:
        - obj_name (str): The name of the detected object.
        - box (tuple): The bounding box of the object, in the format (x, y, w, h).
        - confidence (float): The confidence score of the detection.
    border (int, optional): The number of pixels to ignore from each edge of the image (default is 0).
    percentile (int, optional): The percentile value to use for distance estimation (default is 20).
    confidence_threshold (float, optional): Minimum confidence score to consider a detection (default is 0.6).

    Returns:
    list of tuples: A list containing tuples with:
        - obj_name (str): The name of the detected object.
        - obj_dist (float): The estimated distance to the detected object.
    """
    object_distances = []
    height, width = distances.shape[:2]

    for obj in detected_objects:
        obj_name, box, confidence = obj

        # Skip detections below the confidence threshold
        if confidence < confidence_threshold:
            continue

        # Extract bounding box coordinates and ensure they are within bounds
        x, y, w, h = map(int, box[0])

        # Clamp the bounding box to ensure it stays within the valid area, considering the border
        x = max(border, x)
        y = max(border, y)
        w = min(width - border - x, w)
        h = min(height - border - y, h)

        # Ensure the width and height are positive
        if w <= 0 or h <= 0:
            continue

        # Calculate the object's distance from the depth map
        obj_dist = calculate_object_distance(distances, x, y, w, h, percentile)

        # If a valid distance is found, add it to the result list
        if obj_dist is not None:
            object_distances.append((obj_name, obj_dist))

    return object_distances

def calculate_object_distance(distances_map, x, y, w, h, percentile=20):
    """
    Calculate the distance to an object using the depth values within the bounding box region.

    Parameters:
    distances_map (numpy.ndarray): A depth map containing distance values for each pixel.
    x (int): The x-coordinate of the top-left corner of the bounding box.
    y (int): The y-coordinate of the top-left corner of the bounding box.
    w (int): The width of the bounding box.
    h (int): The height of the bounding box.
    percentile (int, optional): The percentile value to use for distance estimation (default is 20).

    Returns:
    float or None: The estimated distance to the object, or None if no valid depth values are found.
    """
    # Shrink the bounding box slightly to reduce background influence (optional step)
    padding = 5  # Reduce the box by 5 pixels on all sides
    x = max(0, x + padding)
    y = max(0, y + padding)
    w = max(0, w - 2 * padding)
    h = max(0, h - 2 * padding)

    # Ensure that the bounding box is still valid after applying padding
    if w <= 0 or h <= 0:
        return None

    # Get the depth values for the bounding box region
    depth_values = distances_map[y:y + h, x:x + w]
    
    # Ensure that we have valid depth values to work with
    if depth_values.size == 0:
        return None

    # Filter out invalid depth values (e.g., NaN, Inf, or zero values)
    valid_depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0)]

    # If no valid depth values remain, return None
    if valid_depth_values.size == 0:
        return None

    # Use a low percentile to reduce the influence of noise and background
    low_percentile_value = np.percentile(valid_depth_values, percentile)

    return low_percentile_value
