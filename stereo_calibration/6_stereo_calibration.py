
# ### This file is part of a project licensed under the Apache License, Version 2.0.
# ### See the LICENSE file for details.


# # Stereo Camera Calibration
# 
# <a href="https://colab.research.google.com/github/ChristianOrr/stereo-camera-calibration/blob/main/stereo_calibration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# This notebook shows how to perform a calibration on a stereo camera pair. You will need left and right stereo images of a checkerboard to run this notebook. The stereo images will need to be split and placed in seperate folders for left and right images and the numbering of the pair needs to match. Make sure to follow the "tips for a successful calibration" section below before capturing the images. The calibration will not always go smoothly so make sure to follow the tips in the notebook and learn more about how calibration works. I recommend watching this <a href="https://youtube.com/playlist?list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo">playlist on camera calibration</a> to understand the theory behind it. 


# ## Tips for a Successful Calibration
# 
# - Good Lighting is important when capturing the checkerboard images. Make sure there is enough light, but also not too much direct light on the board as this may cause reflections.
# - Print the checkerboard using a high quality printer from a print shop, dont use a home printer.
# - Print on flat and rigid material. The board must not bend while capturing the images.
# - Larger boards are better, I recommend using an A1 size.
# - Take lots of images. Minimum of 20, but dont got more than 100 as processing time will be bad.
# - Capture the board at different angles, rotations and depths from the stereo camera.
# - Make sure the cameras are properly focused, so that there is no bluring on the board.
# - Dont move the camera of board too quickly while capturing images to minimize motion blur.

# %%
import os
import json
from datetime import datetime
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)



# ### Initialise Image Paths and Resolution
# 
# Update the paths and image resolution below.

# %%
#environment_name = 'gym_mountaincar'  # @param ['dm_cartpole', 'gym_mountaincar']
left_dir = "../data/stereo_images/scenes/left/" # @param {type:"string"}
right_dir = "../data/stereo_images/scenes/right/" # @param {type:"string"}
corners_dir = "../data/stereo_images/scenes/corners" # @param {type:"string"}
calibration_dir = "../data/stereo_images/scenes/calibration_results" # @param {type:"string"}

# make folders for corners detections and calibration results 
os.makedirs(corners_dir, exist_ok=True)
os.makedirs(calibration_dir, exist_ok=True)

# Extract image paths and sort so left and right images match
left_image_names = sorted([name for name in os.listdir(left_dir) if os.path.isfile(f"{left_dir}/{name}")])
right_image_names = sorted([name for name in os.listdir(right_dir) if os.path.isfile(f"{right_dir}/{name}")])

num_images = len(left_image_names) # @param {type: "integer"}
image_width = 640 # @param {type: "integer"}
image_height = 480 # @param {type: "integer"}


# ### Set Checkerboard Corners for Object Coordinate System  
# 
# The corners of the checkerboard only refer to the inner corners of the board. The 3D corners are used to calculate the rotation and translation vectors of corners in the camera coordinate systems left and right cameras with respect to the objects coordinate system (checkerboard coordinate system). 
# 
# Make sure to count the number of corner rows and columns.Then update the num_vertical_corners and num_horizontal_corners with the correct values below.

# %%
num_vertical_corners = 6 # @param {type: "integer"}
num_horizontal_corners = 9 # @param {type: "integer"}
total_corners = num_vertical_corners * num_horizontal_corners
objp = np.zeros((num_vertical_corners * num_horizontal_corners, 1, 3), np.float32)
objp[:,0, :2] = np.mgrid[0:num_vertical_corners, 0:num_horizontal_corners].T.reshape(-1, 2)
objp = np.array([corner for [corner] in objp])


# ### Create a Dataframe for the Calibration Results

# %%
calib_dict = {
	"image_id": [i for i in range(1, num_images + 1)],
	"left_image_name": left_image_names,
	"right_image_name": right_image_names,
	"found_chessboard": True, # True or False
	"left_corners": "", # 2d points of chessboard corners in image plane
	"right_corners": "", # 2d points of chessboard corners in image plane
	"object_points": [[objp] for i in range(1, num_images + 1)], # 3d point in real world space (for left and right images)
	"left_rotations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"right_rotations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"left_translations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"right_translations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"left_reprojection_errors": "", # [x, y] error per corner in pixels
	"right_reprojection_errors": "", # [x, y] error per corner in pixels
	"left_error": "", # similar to rms, but per image
	"right_error": "", # similar to rms, but per image
	"left_reprojection_points": "", # [x, y] error per corner in pixels
	"right_reprojection_points": "" # [x, y] error per corner in pixels
}

calib_df = pd.DataFrame(calib_dict)
calib_df = calib_df.set_index("image_id")


# ## Search for checkerboard Corners
# 
# The OpenCV method findChessboardCornersSB is used over the standard findChessboardCorners method since its more accurate. Any image pair with failed corner detections will automatically be removed at the end of searching, so dont be surprised if there are image pairs missing after this phase.

# %%
for image_id, row in calib_df.iterrows():
	print ('\nProcessing Left Image: {}, Right Image: {}'.format(row["left_image_name"], row["right_image_name"]))
	left_image = cv2.imread("{}/{}".format(left_dir, row["left_image_name"]), 1)
	left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
	right_image = cv2.imread("{}/{}".format(right_dir, row["right_image_name"]), 1)
	right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)	  
	# Search for corners
	checkerboard_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
	left_retval, left_corners_array = cv2.findChessboardCornersSB(left_gray, (num_vertical_corners, num_horizontal_corners), checkerboard_flags)
	right_retval, right_corners_array = cv2.findChessboardCornersSB(right_gray, (num_vertical_corners, num_horizontal_corners), checkerboard_flags)

	if not (left_retval and right_retval):
		calib_df["found_chessboard"].loc[image_id] = False
		print("Failed to find corners!")
		continue

	left_corners_array = left_corners_array.reshape(-1,2)
	right_corners_array = right_corners_array.reshape(-1,2)

	calib_df["left_corners"].loc[image_id] = left_corners_array
	calib_df["right_corners"].loc[image_id] = right_corners_array
	calib_df["object_points"].loc[image_id] = objp

	# Save detected corners
	left_corners_image = cv2.drawChessboardCorners(left_image, (num_vertical_corners, num_horizontal_corners), left_corners_array, left_retval)
	cv2.imwrite("{}/{}".format(corners_dir, row["left_image_name"]), left_corners_image)
	right_corners_image = cv2.drawChessboardCorners(right_image, (num_vertical_corners, num_horizontal_corners), right_corners_array, right_retval)
	cv2.imwrite("{}/{}".format(corners_dir, row["right_image_name"]), right_corners_image) 

# Remove all images that failed to detect the corners
calib_df.drop(calib_df[(calib_df["found_chessboard"] == False)].index, axis=0, inplace=True)


# ### Left and Right Detected Corners
# 
# The left image (blue dots) should be on the left side and right image (orange dots) on the right side. 
# 
# Note: The rows may not be perfectly inline. This isn't something to be concerned about since it will be solved later with the stereo calibration.

# %%
index = 0 # @param {type: "integer"}

left_X = calib_df["left_corners"].iloc[index][:,0]
left_Y = calib_df["left_corners"].iloc[index][:,1]
right_X = calib_df["right_corners"].iloc[index][:,0]
right_Y = calib_df["right_corners"].iloc[index][:,1]

# ## Find Intrinsics
# 
# We need to find the intrinsics (pixel focal lengths and principal points) for the left and right cameras. We also need to find the distortion coefficients of the cameras, which will be barrel distortion if a wide angle lenses are used. 
# 
# We will use the detected checkerboard points found above together with the known object coordinate system to help us find the intrinsics and distortion.


# ### Create dictionary to save calibration results

# %%
results_dict = {
	"left_camera_matrix": "",
	"right_camera_matrix": "",
	"left_dist": "",
	"right_dist": "",
	"width": image_width,
	"height": image_height,
	"DIM": (image_width, image_height),
	"left_rms": "",
	"right_rms": "",	
	"left_map_x_undistort": "", 
	"left_map_y_undistort": "",
	"right_map_x_undistort": "", 
	"right_map_y_undistort": "",
	"left_right_flags": cv2.CALIB_ZERO_TANGENT_DIST,
	"left_right_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
	# Stereo values
	"stereo_rms": "",
	"R": "", 
	"T": "",
	"stereo_flags": cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_ZERO_TANGENT_DIST,
	"stereo_criteria": (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
	"left_RT": "", 
	"right_RT": "", 
	"left_P": "", 
	"right_P": "", 
	"Q": "",
	"left_map_x_rectify": "", 
	"left_map_y_rectify": "",
	"right_map_x_rectify": "", 
	"right_map_y_rectify": "",
}


# ### Find Left Intrinsic Matrix and Distortion Coefficients
# 
# A good calibration will have an RMS below 0.4 and the left camera reprojection error plot should show a circular cluster centered around 0.

# %%
def calibrate_left(calib_df, results_dict):
	results_dict["left_rms"], results_dict["left_camera_matrix"], results_dict["left_dist"], left_rotations, left_translations = cv2.calibrateCamera(
		objectPoints = calib_df["object_points"].to_numpy(), 
		imagePoints = calib_df["left_corners"].to_numpy(), 
		imageSize = results_dict["DIM"],
		cameraMatrix = np.zeros((3, 3)),
		distCoeffs = np.zeros((4, 1)),
		rvecs = calib_df["left_rotations"].to_numpy(),
		tvecs = calib_df["left_translations"].to_numpy(),
		flags = results_dict["left_right_flags"]
		)


	# Update df with the calibration results
	calib_df["left_rotations"].loc[list(calib_df.index)] = left_rotations
	calib_df["left_translations"].loc[list(calib_df.index)] = left_translations

	# Display Calibration Performance
	print("\nLeft Camera RMS: {}".format(results_dict["left_rms"]))
	print("\nLeft Camera Matrix: \n{}".format(results_dict["left_camera_matrix"] ))
	print("\nLeft Distortion Coefficients: \n{}\n".format(results_dict["left_dist"]))

	# Reprojection error
	for image_id, row in calib_df.iterrows():
		left_reprojected_points, _ = cv2.projectPoints(
			objectPoints = row["object_points"], 
			rvec = row["left_rotations"], 
			tvec = row["left_translations"], 
			cameraMatrix = results_dict["left_camera_matrix"] , 
			distCoeffs = results_dict["left_dist"]
			)
		# Add reprojection errors and points to the dataframe
		left_reprojected_points = left_reprojected_points.reshape(-1,2)
		calib_df["left_reprojection_errors"].loc[image_id] = row["left_corners"] - left_reprojected_points
		calib_df["left_reprojection_points"].loc[image_id] = left_reprojected_points
		# find error similar to rms per image
		calib_df["left_error"].loc[image_id] = np.sqrt(np.sum(np.square(row["left_corners"] - left_reprojected_points)) / total_corners)

	# Plot the reprojection errors
	left_errors = np.stack(calib_df["left_reprojection_errors"].to_numpy())
	return calib_df, results_dict
calib_df, results_dict = calibrate_left(calib_df, results_dict)


# ### Find Right Intrinsic Matrix and Distortion Coefficients
# 
# The metrics for a good calibration are the same as the left image. The RMS should be below 0.4 and the left camera reprojection error plot should show a circular cluster centered around 0.

# %%
def calibrate_right(calib_df, results_dict):
	results_dict["right_rms"], results_dict["right_camera_matrix"], results_dict["right_dist"], right_rotations, right_translations = cv2.calibrateCamera(
		objectPoints = calib_df["object_points"].to_numpy(), 
		imagePoints = calib_df["right_corners"].to_numpy(), 
		imageSize = results_dict["DIM"],
		cameraMatrix = np.zeros((3, 3)),
		distCoeffs = np.zeros((4, 1)),
		rvecs = calib_df["right_rotations"].to_numpy(),
		tvecs = calib_df["right_translations"].to_numpy(),
		flags = results_dict["left_right_flags"]
		)

	# Update all rows with the calibration values
	calib_df["right_rotations"].loc[list(calib_df.index)] = right_rotations
	calib_df["right_translations"].loc[list(calib_df.index)] = right_translations

	# Display Calibration Results
	print("\nRight Camera RMS: {}".format(results_dict["right_rms"]))
	print("\nRight Camera Matrix: \n{}".format(results_dict["right_camera_matrix"]))
	print("\nRight Distortion Coefficients: \n{}\n".format(results_dict["right_dist"]))

	# Reprojection error
	for image_id, row in calib_df.iterrows():
		right_reprojected_points, _ = cv2.projectPoints(
			objectPoints = row["object_points"], 
			rvec = row["right_rotations"], 
			tvec = row["right_translations"], 
			cameraMatrix = results_dict["right_camera_matrix"], 
			distCoeffs = results_dict["right_dist"]
			)
		# Add reprojection errors and points to the dataframe
		right_reprojected_points = right_reprojected_points.reshape(-1,2)
		calib_df["right_reprojection_errors"].loc[image_id] = row["right_corners"] - right_reprojected_points
		calib_df["right_reprojection_points"].loc[image_id] = right_reprojected_points
		# find error similar to rms per image
		calib_df["right_error"].loc[image_id] = np.sqrt(np.sum(np.square(row["right_corners"] - right_reprojected_points)) / total_corners)

	# Plot the reprojection errors
	right_errors = np.stack(calib_df["right_reprojection_errors"].to_numpy())
	return calib_df, results_dict
calib_df, results_dict = calibrate_right(calib_df, results_dict)



# Below is the ID's for each image pair for reference

# %%
calib_df[["left_image_name", "right_image_name"]]


# ### Plot reprojected over detected corners
# 
# If calibration is successful, then none of the detected corners (blue dots) should be visible. This validates that the calibration was successful.

# %%
index = 0 # Select the image to display

left_X = calib_df["left_corners"].iloc[index][:,0]
left_Y = calib_df["left_corners"].iloc[index][:,1]
lproj_X = calib_df["left_reprojection_points"].iloc[index][:,0] 
lproj_Y = calib_df["left_reprojection_points"].iloc[index][:,1]

right_X = calib_df["right_corners"].iloc[index][:,0]
right_Y = calib_df["right_corners"].iloc[index][:,1]
rproj_X = calib_df["right_reprojection_points"].iloc[index][:,0] 
rproj_Y = calib_df["right_reprojection_points"].iloc[index][:,1]


# ### Remove Worst Images and Recalibrate if Necessary
# 
# If the calibration did not perform well, then remove the highest RMS images and recalibrate. 
# You can tell if the calibration was poor if for example the blue dots are not fully covered by the orange dots in the reprojected over detected images above. 
# Another sign of poor performance is if the RMS is high. 0.4 or greater RMS is a high value.
# 
# Its generally better to have more images, but if there is a large batch of bad images then you can remove more images by decreasing keep_best_ratio.

# %%
recalibrate = True # @param {type:"boolean"}
# Keep the images with top rms performace (lower ratio means fewer images)
keep_best_ratio = 0.7 # @param {type:"slider", min:0, max:1, step:0.01} 

if recalibrate:
    # Find and drop images with the worst performance
    mean_errors = calib_df["mean_errors"]
    mean_errors = mean_errors.sort_values()

    # select worst images to remove
    images_to_drop = list(mean_errors.iloc[int(len(mean_errors) * keep_best_ratio):].index)
    # Remove poor performing images
    calib_df.drop(images_to_drop, axis=0, inplace=True)
    print("\nLeft recalibration")
    calib_df, results_dict = calibrate_left(calib_df, results_dict)
    print("\nRight recalibration")
    calib_df, results_dict = calibrate_right(calib_df, results_dict)


# ## Find Extrinsics
# 
# Now that the intrinsics and distortion coefficients have been calculated for the left and right cameras, we can calculate the extrinsics for the camera coordinate systems with respect to each other.


# ### Rotation and Translation of Camera Coordinate Systems
# 
# Find the rotation, R and translation T of the left camera coordinate system with respect to the right camera coordinate system.

# %%
results_dict["stereo_rms"], _, _, _, _, results_dict["R"], results_dict["T"], E, F = cv2.stereoCalibrate(
	objectPoints = calib_df["object_points"].to_numpy(), 
	imagePoints1 = calib_df["left_corners"].to_numpy(), 
	imagePoints2 = calib_df["right_corners"].to_numpy(), 
	cameraMatrix1 = results_dict["left_camera_matrix"], 
	distCoeffs1 = results_dict["left_dist"], 
	cameraMatrix2 = results_dict["right_camera_matrix"], 
	distCoeffs2 = results_dict["right_dist"], 
	imageSize = results_dict["DIM"], 
	R = None, 
	T = None,
	flags = results_dict["stereo_flags"], 
	criteria = results_dict["stereo_criteria"]
	)

print ("Stereo RMS: ", results_dict["stereo_rms"])


# ### Left and Right Rectification Transforms and Projection Matrices
# 
# Find the left rectification transform, left_RT which is used to transform the points in the unrectified left cameras coordinate system to the rectified left camera coordinate system. Similarly the right rectification transform, right_RT is used to transform the points in the unrectified right cameras coordinate system to the rectified right camera coordinate system.
# 
# The left projection matrix is used to project points in the rectified left camera coordinate system (obtained using left_RT) to the rectified left camera's image. Similarly the right projection matrix is used to project points in the rectified right camera coordinate system (obtained using right_RT) to the rectified right camera's image.

# %%
(results_dict["left_RT"], results_dict["right_RT"], results_dict["left_P"], results_dict["right_P"], results_dict["Q"], validPixROI1, validPixROI2) = cv2.stereoRectify(
	cameraMatrix1 = results_dict["left_camera_matrix"], 
	distCoeffs1 = results_dict["left_dist"], 
	cameraMatrix2 = results_dict["right_camera_matrix"], 
	distCoeffs2 = results_dict["right_dist"], 
	imageSize = results_dict["DIM"], 
	R = results_dict["R"], 
	T = results_dict["T"]
	)


# ## Save Results for Reuse
# 
# Now that all the necessary calibration information has been extracted the results can be saved for reuse. These calibration results will be valid as long as the stereo camera's position with respect to eachother doesnt move and the cameras sensors or lenses dont get changed. In the case that any of these changes are made the calibration will need to be performed again. 
# 
# The stereo images can now be rectified and used for disparity estimation using stereo matching techniques.


# ### Undistortion Maps
# 
# The left and right maps below can be used for removing distortion from the images. The maps are built for the inverse mapping algorithm used by the OpenCV remap function to translate the pixel coordinates into the correct undistorted position. The x maps translates the pixels in the x direction and the y map translates the pixels in the y direction.
# 
# Note: These maps are not needed for disparity estimation, since they don't rectify the left and right images with respect to each other. See below for the rectification maps.

# %%

results_dict["left_map_x_undistort"], results_dict["left_map_y_undistort"] = cv2.initUndistortRectifyMap(
	cameraMatrix = results_dict["left_camera_matrix"], 
	distCoeffs = results_dict["left_dist"], 
	R = None, 
	newCameraMatrix = None, 
	size = results_dict["DIM"], 
	m1type = cv2.CV_16SC2)
results_dict["right_map_x_undistort"], results_dict["right_map_y_undistort"] = cv2.initUndistortRectifyMap(
	cameraMatrix = results_dict["right_camera_matrix"], 
	distCoeffs = results_dict["right_dist"], 
	R = None, 
	newCameraMatrix = None, 
	size = results_dict["DIM"], 
	m1type = cv2.CV_16SC2)

np.savez(
	"{}/calibration_left.npz".format(calibration_dir),
	left_map = results_dict["left_map_x_undistort"], 
	right_map = results_dict["left_map_y_undistort"], 
	objpoints = calib_df["object_points"].to_numpy(), 
	imgpoints = calib_df["left_corners"].to_numpy(),
	camera_matrix = results_dict["left_camera_matrix"], 
	distortion_coeff = results_dict["left_dist"], 
	imageSize = results_dict["DIM"]
	)
np.savez(
	"{}/calibration_right.npz".format(calibration_dir),
	left_map = results_dict["right_map_x_undistort"], 
	right_map = results_dict["right_map_y_undistort"], 
	objpoints = calib_df["object_points"].to_numpy(), 
	imgpoints = calib_df["right_corners"].to_numpy(),
	camera_matrix = results_dict["right_camera_matrix"], 
	distortion_coeff = results_dict["right_dist"], 
	imageSize = results_dict["DIM"]
	)


# ### Rectification Maps
# 
# The maps below can be used to undistort and rectify the stereo images. The maps are built for the inverse mapping algorithm used by the OpenCV remap function to translate the pixel coordinates into the correct rectified position. The x maps translates the pixels in the x direction and the y map translates the pixels in the y direction.

# %%
results_dict["left_map_x_rectify"], results_dict["left_map_y_rectify"] = cv2.initUndistortRectifyMap(
	cameraMatrix = results_dict["left_camera_matrix"], 
	distCoeffs = results_dict["left_dist"], 
	R = results_dict["left_RT"], 
	newCameraMatrix = results_dict["left_P"], 
	size = results_dict["DIM"], 
	m1type = cv2.CV_16SC2 
	)
results_dict["right_map_x_rectify"], results_dict["right_map_y_rectify"] = cv2.initUndistortRectifyMap(
	cameraMatrix = results_dict["right_camera_matrix"], 
	distCoeffs = results_dict["right_dist"], 
	R = results_dict["right_RT"], 
	newCameraMatrix = results_dict["right_P"], 
	size = results_dict["DIM"], 
	m1type = cv2.CV_16SC2 
)

np.savez_compressed(
	calibration_dir + "/stereo_calibration.npz", 
	imageSize = results_dict["DIM"],
	left_map_x_rectify = results_dict["left_map_x_rectify"], 
	left_map_y_rectify = results_dict["left_map_y_rectify"],
	right_map_x_rectify = results_dict["right_map_x_rectify"], 
	right_map_y_rectify = results_dict["right_map_y_rectify"], 
	disparityToDepthMap = results_dict["Q"],
	rotationMatrix = results_dict["R"], 
	translationVector = results_dict["T"]
	)

results_output_dict = {
	"left_camera_matrix": results_dict["left_camera_matrix"].tolist(),
	"right_camera_matrix": results_dict["right_camera_matrix"].tolist(),
	"left_dist": results_dict["left_dist"].tolist(),
	"right_dist": results_dict["right_dist"].tolist(),
	"width": results_dict["width"],
	"height": results_dict["height"],
	"DIM": results_dict["DIM"],
	"left_rms": results_dict["left_rms"],
	"right_rms": results_dict["right_rms"],	
	"left_right_flags": results_dict["left_right_flags"],
	"left_right_criteria": results_dict["left_right_criteria"],
	# Stereo values
	"stereo_rms": results_dict["stereo_rms"],
	"R": results_dict["R"].tolist(), 
	"T": results_dict["T"].tolist(),
	"stereo_flags": results_dict["stereo_flags"],
	"stereo_criteria": results_dict["stereo_criteria"],
	"left_RT": results_dict["left_RT"].tolist(), 
	"right_RT": results_dict["right_RT"].tolist(), 
	"left_P": results_dict["left_P"].tolist(), 
	"right_P": results_dict["right_P"].tolist(), 
	"Q": results_dict["Q"].tolist(),
	"calibration_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
}

results_path = "{}/calibration_results.json".format(calibration_dir)
with open(results_path, "w") as file:
	json.dump(results_output_dict, file, indent=4, sort_keys=False)


