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
    
    # undistortion maps
    left_map_x_undistort = left_npzfile["left_map"]
    right_map_x_undistort = right_npzfile["left_map"]
    left_map_y_undistort = left_npzfile["right_map"]
    right_map_y_undistort = right_npzfile["right_map"]
   
    # rectification maps
    left_map_x_rectify = stereo_npzfile["left_map_x_rectify"]
    left_map_y_rectify = stereo_npzfile["left_map_y_rectify"]
    right_map_x_rectify = stereo_npzfile["right_map_x_rectify"]
    right_map_y_rectify = stereo_npzfile["right_map_y_rectify"]
    
    Q = stereo_npzfile["disparityToDepthMap"]
    
    focal_length = left_npzfile['camera_matrix'][0][0]


    # Apply undistortion maps to get undistorted images
    left_undistorted = cv2.remap(left, left_map_x_undistort, left_map_y_undistort, interpolation=cv2.INTER_LINEAR)
    right_undistorted = cv2.remap(right, right_map_x_undistort, right_map_y_undistort, interpolation=cv2.INTER_LINEAR)
    
    # Apply rectification maps (from calibration results) to get rectified images
    left_rectified = cv2.remap(left_undistorted, left_map_x_rectify, left_map_y_rectify, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_undistorted, right_map_x_rectify, right_map_y_rectify, interpolation=cv2.INTER_LINEAR)
    
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

def get_closest_distance(distances, disparity_map, min_thresh=1.6, max_thresh=5, border=20, topBorder=100, min_region_area=2000):
    """
    Find the closest distance by analyzing regions in the disparity map.
    
    Parameters:
        distances (numpy.ndarray): Depth map containing distance values
        disparity_map (numpy.ndarray): Raw disparity map
        min_thresh (float): Minimum valid distance
        max_thresh (float): Maximum valid distance
        border (int): Border size to ignore on left, right, and bottom
        topBorder (int): Border size to ignore from the top
        min_region_area (int): Minimum area (in pixels) for a region to be considered
        
    Returns:
        tuple: (closest_distance, closest_coordinates) or (None, None) if no valid regions found
    """
    # Create initial mask for valid distances
    valid_mask = (
        (distances >= min_thresh) & 
        (distances <= max_thresh) & 
        (distances > 0) &
        np.isfinite(distances)
    )
    
    # Convert disparity map to uint8 for contour detection
    disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply masks with different borders
    height, width = distances.shape
    # Top border
    disparity_normalized[:topBorder, :] = 0
    # Bottom border
    disparity_normalized[-border:, :] = 0
    # Left and right borders
    disparity_normalized[:, :border] = 0
    disparity_normalized[:, -border:] = 0
    
    # Apply threshold to get regions with significant disparity
    _, binary_mask = cv2.threshold(disparity_normalized, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    closest_distance = float('inf')
    closest_coordinates = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_region_area:
            continue
            
        # Get the centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Skip if centroid is in the top border area
        if cY < topBorder:
            continue
        
        # Get region around centroid
        window_size = 15
        half_window = window_size // 2
        
        # Ensure coordinates are within bounds
        start_y = max(0, cY - half_window)
        end_y = min(height, cY + half_window + 1)
        start_x = max(0, cX - half_window)
        end_x = min(width, cX + half_window + 1)
        
        # Get window of distances
        distance_window = distances[start_y:end_y, start_x:end_x]
        valid_window = valid_mask[start_y:end_y, start_x:end_x]
        
        # Calculate average distance of valid points in window
        valid_distances = distance_window[valid_window]
        if len(valid_distances) > 0:
            avg_distance = np.mean(valid_distances)
            if avg_distance < closest_distance:
                closest_distance = avg_distance
                closest_coordinates = (cX, cY)
    
    if closest_coordinates is None:
        return None, None
        
    return closest_distance, closest_coordinates


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
