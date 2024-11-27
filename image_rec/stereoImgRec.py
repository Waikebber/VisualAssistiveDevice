import cv2
import numpy as np

def create_detection_image(distance_map, detected_objects):
    """
    Creates an image with the detected objects' bounding boxes, labels, and confidence scores overlaid on the distance map.
    
    Args:
        distance_map (numpy.ndarray): The depth map representing distances.
        detected_objects (list of tuples): Each tuple contains (object_name, confidence, (x_min, y_min, x_max, y_max)).

    Returns:
        numpy.ndarray: Image with bounding boxes and labels.
    """
    # Convert the depth map to a colored representation
    depth_display_colored = cv2.applyColorMap(cv2.convertScaleAbs(distance_map[:, :, 2], alpha=255.0 / np.max(distance_map[:, :, 2])), cv2.COLORMAP_JET)
    
    # Draw bounding boxes and labels on the depth map
    for obj_name, confidence, (x_min, y_min, x_max, y_max) in detected_objects:
        # Draw bounding box
        cv2.rectangle(depth_display_colored, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Create label with object name and confidence
        label = f"{obj_name}: {confidence * 100:.1f}%"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_left = (x_min, y_min - label_size[1] - base_line)
        bottom_right = (x_min + label_size[0], y_min)
        cv2.rectangle(depth_display_colored, top_left, bottom_right, (0, 255, 0), cv2.FILLED)
        cv2.putText(depth_display_colored, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return depth_display_colored

def calculate_object_distances(distance_map, detected_objects):
    """
    Calculates the distance of objects within the bounding boxes.
    
    Args:
        distance_map (numpy.ndarray): The depth map representing distances.
        detected_objects (list of tuples): Each tuple contains (object_name, confidence, (x_min, y_min, x_max, y_max)).
    
    Returns:
        list of tuples: Each tuple contains (object_name, distance_val, confidence, (x_min, y_min, x_max, y_max)).
    """
    object_distances = []
    
    for obj_name, confidence, (x_min, y_min, x_max, y_max) in detected_objects:
        # Extract the region of interest (ROI) from the distance map
        roi = distance_map[y_min:y_max, x_min:x_max, 2]
        
        # Calculate the average distance within the bounding box, ignoring invalid values (e.g., 0 or negative values)
        valid_distances = roi[roi > 0]
        
        if len(valid_distances) > 0:
            distance_val = np.mean(valid_distances)
        else:
            distance_val = float('inf')  # If no valid distances, set to infinity (object not detectable)
        
        # Append the calculated distance along with other details to the result list
        object_distances.append((obj_name, distance_val, confidence, (x_min, y_min, x_max, y_max)))
    
    return object_distances
