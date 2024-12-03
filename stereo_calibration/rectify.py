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

def get_closest_distance(distances, min_thresh=1.6, max_thresh=5, border=20):
    """
    A faster version of get_closest_distance that avoids sliding windows.
    
    Parameters:
        distances (numpy.ndarray): A depth map containing distance values for each pixel
        min_thresh (float): Minimum valid distance
        max_thresh (float): Maximum valid distance
        border (int): Border size to ignore
        
    Returns:
        tuple: (closest_distance, closest_coordinates) or (None, None) if no valid points found
    """
    # Create mask for valid distances
    valid_mask = (
        (distances >= min_thresh) & 
        (distances <= max_thresh) & 
        np.isfinite(distances)
    )
    
    # Apply border mask
    height, width = distances.shape
    valid_mask[:border, :] = False
    valid_mask[-border:, :] = False
    valid_mask[:, :border] = False
    valid_mask[:, -border:] = False
    
    # Find valid points
    valid_points = np.where(valid_mask)
    
    if len(valid_points[0]) == 0:
        return None, None
        
    # Get distances for valid points
    valid_distances = distances[valid_points]
    
    # Find index of minimum distance
    min_idx = np.argmin(valid_distances)
    
    # Get coordinates and distance
    min_y = valid_points[0][min_idx]
    min_x = valid_points[1][min_idx]
    min_distance = valid_distances[min_idx]
    
    # Optional: Verify this isn't an isolated point
    # Get 3x3 window around minimum point
    y_start = max(0, min_y - 1)
    y_end = min(height, min_y + 2)
    x_start = max(0, min_x - 1)
    x_end = min(width, min_x + 2)
    
    window = valid_mask[y_start:y_end, x_start:x_end]
    valid_neighbors = np.sum(window) - 1  # Subtract 1 to not count the point itself
    
    # Only return point if it has at least 2 valid neighbors
    if valid_neighbors >= 2:
        return min_distance, (min_x, min_y)
    
    return None, None


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