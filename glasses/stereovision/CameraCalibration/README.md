# Camera Calibration Explanation
This project performs camera calibration using OpenCV. Calibration is essential for removing distortion from camera lenses, especially for stereo vision setups.<br>
For a detailed explanation of the calibration process, refer to the source blog here. Much of this is derived from a [YouTube video](https://www.youtube.com/watch?v=uKDAVcSaNZA).

## Chessboard
A chessboard, or checkerboard, is used for camera calibration. An initial image of the chessboard is captured, which allows for a correlation between real-world 3D coordinates and their corresponding 2D points in the image. By identifying the corners of the chessboard in multiple images, the camera's intrinsic and extrinsic parameters can be calculated.

## /images/
This directory stores your images of the chessboard that are used for calibration. The script processes the images in this directory to identify the coordinates of the box corners in each image. It contains two subdirectories:
- stereoLeft: Stores images captured from the left camera.
- stereoRight: Stores images captured from the right camera.
- 
These images are processed to determine the camera's distortion and projection matrices.

## Scripts
### ```cameraCalibration.py```
This script handles the calibration of a single camera using chessboard images. It identifies chessboard corners, computes the camera matrix and distortion coefficients, and then saves this calibration data to an XML file called ```stereoMap.xml```.
Steps include:
1. Reading chessboard images.
2. Detecting chessboard corners in each image.
3. Performing the calibration to calculate camera parameters.
4. Saving camera calibration parameter for stereovision.

### ```calibration.py```
This script includes functions for undistorting and rectifying the images based on the calibration metrics obtained earlier. The main function, ```undistortRectify(frameR, frameL)```, takes in stereo images (left and right) and corrects distortion using the calibration data stored in ```stereoMap.xml```. After rectification, the stereo images are ready for 3D depth estimation.

---

For more detailed explanations of camera calibration, you can visit the [OpenCV Camera Calibration Blog](https://learnopencv.com/camera-calibration-using-opencv/).
