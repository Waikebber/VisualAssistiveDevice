# Stereovision Calibration

This section provides a detailed guide to calibrating stereovision cameras, including testing, capturing calibration images, and verifying the quality of the calibration process.

## Steps

1. Test

    This step runs the cameras in a live video feed, ensuring both cameras are properly connected and functioning as expected before moving forward.

2. Chess Cycle

    In this step, the script takes n photos for stereovision calibration. Each photo must include a visible chessboard of the same size in both frames. The left and right images are saved in the same file, stacked horizontally.

3. Split Image Pairs

    This step splits the saved images from the Chess Cycle step into separate left and right images and places them in the appropriate directories for calibration.

4. Check Images

    This step allows you to visually verify the quality of the calibration images. It displays the detected chessboard corners in the images. You should ensure that OpenCV properly detects all inner corners of the chessboard. Press Enter to keep the image pair or Backspace to delete it. Aim to have between 30 and 50 valid images.

5. Fix Directory

    Use this script to properly fix the numbering of image files after deleting some during the previous step, ensuring that the images are numbered sequentially without gaps.

## 6 Stereo Camera Calibration

### Introduction

This step involves calibrating stereo cameras using multiple images of a checkerboard pattern. All the code necessary for calibration is contained in the stereo_calibrate.ipynb notebook. The main goal of calibration is to rectify the stereo images so that they can be used for depth estimation. Proper image capture is essential, so follow the "Tips for a Successful Calibration" section for high-quality results.

### Tips for a Successful Calibration

- Lighting: Ensure good lighting while capturing checkerboard images. Avoid excessive direct light that could cause reflections.

- Checkerboard Quality: Print the checkerboard at a print shop using a high-quality printer. Avoid using a home printer.

- Material: Print on flat, rigid material to prevent bending during image capture.

- Board Size: Use a larger board, ideally A1 size, for better accuracy.

- Number of Images: Capture at least 20 images, but no more than 100 to avoid long processing times.

- Variety in Images: Capture images from different angles, rotations, and depths.

- Focus: Ensure the cameras are focused properly to prevent blurring of the chessboard.

- Minimize Motion Blur: Avoid moving the board or cameras too quickly during capture to reduce motion blur.

### Theory Behind Calibration

For a deeper understanding of the calibration process, refer to this YouTube playlist on camera calibration. The algorithms used for calibration are part of OpenCV, and you can find further technical details in their calibration documentation.

Calibration was Successful, What's Next?

Once the stereo camera is calibrated, you can use it for depth estimation. A good starting point is OpenCV's block matching techniques, which provide basic stereo depth estimation. For more advanced methods, consider exploring deep stereo techniques like MADNet.

### LICENSE

This code is derived from the [stereo-camera-calibration repository](https://github.com/ChristianOrr/stereo-camera-calibration) by ChristianOrr and follows the original license.d backspace to delete the image pair. Make sure that there are 30-50 image that are valid.