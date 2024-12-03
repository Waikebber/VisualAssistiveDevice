"""
For rectifying stereo images and computing disparity maps.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter

def rectify_imgs(left, right, calibration_dir="../data/stereo_images/scenes/calibration_results"):
    """ Rectify stereo images using precomputed calibration data.
    Parameters:
        left (numpy.ndarray): The left image to be rectified.
        right (numpy.ndarray): The right image to be rectified.
        calibration_dir (str): Directory containing the calibration data files. Default is "../data/stereo_images/scenes/calibration_results".
    Returns:
        tuple: A tuple containing:
            - left_rectified (numpy.ndarray): The rectified left image.
            - right_rectified (numpy.ndarray): The rectified right image.
            - Q (numpy.ndarray): The disparity-to-depth mapping matrix.
            - focal_length (float): The focal length of the left camera.
    """
    
    # Step 1: Load Calibration Data and Rectify Images
    # Load calibration data
    left_npzfile = np.load("{}/calibration_left.npz".format(calibration_dir))
    right_npzfile = np.load("{}/calibration_right.npz".format(calibration_dir))
    stereo_npzfile = np.load("{}/stereo_calibration.npz".format(calibration_dir))
   
    # rectification maps
    left_map_x_rectify = stereo_npzfile["left_map_x_rectify"]
    left_map_y_rectify = stereo_npzfile["left_map_y_rectify"]
    right_map_x_rectify = stereo_npzfile["right_map_x_rectify"]
    right_map_y_rectify = stereo_npzfile["right_map_y_rectify"]
    
    Q = stereo_npzfile["disparityToDepthMap"]
    
    focal_length = left_npzfile['camera_matrix'][0][0]

    # Apply rectification maps (from calibration results) to get rectified images
    left_rectified = cv2.remap(left, left_map_x_rectify, left_map_y_rectify, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, right_map_x_rectify, right_map_y_rectify, interpolation=cv2.INTER_LINEAR)
    
    return left_rectified, right_rectified, Q, focal_length

def make_disparity_map(left_rectified, right_rectified, min_disp=0, num_disp=16*9, block_size=9):
    """
    Generates a disparity map from rectified stereo image pairs using the Semi-Global Block Matching (SGBM) algorithm.
    
    Args:
        left_rectified (numpy.ndarray): The rectified left image.
        right_rectified (numpy.ndarray): The rectified right image.
        min_disp (int, optional): Minimum possible disparity value. Default is 0.
        num_disp (int, optional): Maximum disparity minus minimum disparity. Must be divisible by 16. Default is 16*9.
        block_size (int, optional): Matched block size. It must be an odd number >=1. Default is 9.
    Returns:
        numpy.ndarray: The computed disparity map as a floating-point array.
    """
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2
    )

    # Compute disparity map
    disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    
    return disparity_map

def get_closest_distance(distances, min_thresh=1, max_thresh=5, morph_filter=True, window_size=21, border = 20):
    """
    Finds the closest valid distance from the depth map while avoiding noise.

    Parameters:
        distances (numpy.ndarray): A depth map containing distance values for each pixel.
        min_thresh (float, optional): The minimum threshold distance to be considered valid.
        max_thresh (float, optional): The maximum threshold distance to be considered valid.
        morph_filter (bool, optional): Whether to apply morphological filtering to reduce isolated noise.
        window_size (int, optional): The size of the sliding window to evaluate local clusters.
        border (int, optional): The border size to ignore around the edges of the depth map.

    Returns:
        tuple: Closest distance (float) and the coordinates of the closest point (tuple).
    """
    # Create a copy to avoid modifying the original
    distances_copy = distances.copy()
    
    # Filter out invalid depth values
    invalid_mask = (~np.isfinite(distances_copy)) | (distances_copy <= min_thresh) | (distances_copy >= max_thresh)
    distances_copy[invalid_mask] = np.nan
    
    # Apply median filter to reduce noise
    distances_filtered = median_filter(distances_copy, size=3)
    
    # Create validity mask
    valid_mask = np.isfinite(distances_filtered)
    
    if morph_filter:
        kernel = np.ones((3, 3), np.uint8)
        valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the mask
    distances_filtered[valid_mask != 1] = np.nan
    
    # If no valid distances remain, return None
    if not np.any(np.isfinite(distances_filtered)):
        return None, None
    
    # Find clusters of valid points using a sliding window
    height, width = distances_filtered.shape
    min_distance = float('inf')
    min_coordinates = None
    
    for y in range(border, height - border):
        for x in range(border, width - border):
            # Get window around current point
            window = distances_filtered[
                max(0, y - window_size//2):min(height, y + window_size//2 + 1),
                max(0, x - window_size//2):min(width, x + window_size//2 + 1)
            ]
            
            # Check if we have enough valid points in the window
            valid_points = np.sum(np.isfinite(window))
            if valid_points > (window_size * window_size * 0.3):  # At least 30% of window should be valid
                current_distance = distances_filtered[y, x]
                if np.isfinite(current_distance) and current_distance < min_distance:
                    min_distance = current_distance
                    min_coordinates = (x, y)
    
    if min_coordinates is None:
        return None, None
        
    return min_distance, min_coordinates


def make_colored_distance_map(distances, min_distance, max_distance):
    """
    Generates a colored distance map from a given distance array, applying a specified threshold range.
    Parameters:
        distances (numpy.ndarray): A 2D array of distance values.
        min_distance (float): The minimum distance threshold.
        max_distance (float): The maximum distance threshold.
    Returns:
        numpy.ndarray: A 3D array representing the colored distance map, or None if no values are within the threshold range.
    """
    # Create a masked version of distances between MIN_THRESH and MAX_THRESH
    masked_distances = np.where((distances >= min_distance) & (distances <= max_distance), distances, np.nan)

    # Normalize the depth values within the threshold range for display purposes
    valid_mask = np.isfinite(masked_distances)  # Mask to find valid (non-NaN) values
    
    if np.sum(valid_mask) > 0:
        # Normalize only valid values within the range
        min_valid = np.nanmin(masked_distances)
        max_valid = np.nanmax(masked_distances)
        depth_map_normalized = np.zeros_like(masked_distances)
        
        # Apply normalization only to the valid range
        depth_map_normalized[valid_mask] = 255 * (masked_distances[valid_mask] - min_valid) / (max_valid - min_valid)
        depth_map_normalized = depth_map_normalized.astype(np.uint8)

        # Apply a colormap for better visualization
        depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        depth_map_colored = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)
        return depth_map_colored
    else:
        print(f"No values found within the threshold range [{min_distance}, {max_distance}] meters.")
        return None