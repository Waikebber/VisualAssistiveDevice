# Camera Calibration Explanation

This directory contains all the necessary files and images to perform camera calibration using OpenCV. Camera calibration is crucial for removing distortion from images, which occurs due to the lens' curvature. By taking multiple images of a chessboard, we can compute the intrinsic parameters of the camera.

## Chessboard
A chessboard (checkerboard) is used as the object for calibration due to its well-defined geometry. The corner points of the squares serve as reference points to correlate the real-world 3D coordinates with 2D image coordinates. An initial image is captured in front of the board, which helps to establish this correlation.

## /images/
This directory stores all the chessboard images used for calibration. The camera calibration script iterates through these images to detect the corner points of the chessboard in each one. The more images used from different angles, the more accurate the calibration.

## Scripts
### cameraCalibration.py
The original OpenCV code can be found [here](https://github.com/spmallick/learnopencv/blob/master/CameraCalibration/cameraCalibration.py).

This script detects the corner points in the images and computes the camera matrix, distortion coefficients, rotation, and translation vectors. These values are used to model the camera’s intrinsic and extrinsic properties.

Steps include:
1. Reading chessboard images.
2. Detecting chessboard corners in each image.
3. Performing the calibration to calculate camera parameters.

### cameraCalibrationWithUndistortion.py
The original OpenCV code can be found [here](https://github.com/spmallick/learnopencv/blob/master/CameraCalibration/cameraCalibrationWithUndistortion.py).

This script goes a step further by applying the calibration results to undistort the images. It removes the distortion caused by the camera lens, producing geometrically corrected images.

Steps include:
1. Loading the camera calibration parameters (camera matrix and distortion coefficients).
2. Undistorting each image using these parameters.
3. Displaying the original and corrected images side by side.

---

For more detailed explanations of camera calibration, you can visit the [OpenCV Camera Calibration Blog](https://learnopencv.com/camera-calibration-using-opencv/).
