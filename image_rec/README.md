# Stereo Image Recognition Project

This repository implements a stereo image recognition system that uses YOLO for object detection and leverages stereo camera input for advanced processing. The main functionality includes object detection, annotated image creation, and distance estimation based on depth maps.

---

## File Overview

### **`img_rec.py`**
- **Purpose**: Core functionality for image recognition using the YOLO model.
- **Highlights**:
  - `ImgRec` class manages the YOLO model and provides methods for:
    - Predicting objects in a single image or folder.
    - Annotating images with bounding boxes and confidence scores.
    - Configurable confidence thresholds and saving results.
- **Key Methods**:
  - `predict_one_image`: Runs YOLO on a single image.
  - `predict_folder`: Processes all images in a directory.
  - `predict_frame`: Processes a single frame in memory.

### **`stereoImgRec.py`**
- **Purpose**: Helper functions to process stereo images and calculate distances.
- **Highlights**:
  - `create_detection_image`: Draws bounding boxes and labels on detected objects.
  - `calculate_object_distances`: Estimates distances of objects using a depth map.
  - `calculate_object_distance`: Extracts distance information using percentile filtering.

### **`run_img_rec.py`**
- **Purpose**: A CLI script for running image recognition.
- **Usage**:
  - Run on single images or directories.
  - Command-line parameters:
    - `input_path`: Path to the image or folder.
    - `confidence_threshold` (optional): Detection confidence threshold (default: 0.5).
    - `save_result` (optional): Whether to save annotated images (default: False).

### **`imageRecognition.py`**
- **Purpose**: Main program integrating stereo cameras and YOLO-based image recognition.
- **Features**:
  - Initializes stereo cameras using Picamera2.
  - Continuously captures frames from both cameras.
  - Allows user interaction via keyboard:
    - **`p`**: Capture and classify images from both cameras.
    - **`q`**: Quit the application.
  - Annotates and saves images with detected objects.

---

## How to Use

1. **Install Dependencies**:
   - Python 3.x
   - Required libraries:
     ```bash
     pip install opencv-python-headless ultralytics pillow picamera2
     ```

2. **Run the Main Program**:
   ```bash
   python imageRecognition.py
