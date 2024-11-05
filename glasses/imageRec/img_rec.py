import os, glob
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

class ImgRec:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def predict_one_image(self, img_path, save_result=False, save_folder='./predictions', confidence_threshold=0.5):
        """Creates an image recognition prediction for a single image.
           If a save folder is provided, the result will be saved with '_prediction' appended.

        Args:
            img_path (str): Path to the image.
            save_result (bool, optional): Saves the resulting classification to a file. Defaults to False.
            save_folder (str, optional): Folder path to save the result. Defaults to './predictions'.
            confidence_threshold (float, optional): Confidence threshold for detected objects. Defaults to 0.5.

        Returns:
            list: List of detected objects. Each object is a tuple containing the object name, bounding box, and confidence.
        """
        img = Image.open(img_path)
        results = self.model(img)

        # Process the results to extract detected objects and their information
        interpreted_results = self._interpret_results(results, confidence_threshold)

        # Ensure save folder exists if saving results
        if save_result:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, os.path.basename(img_path).replace('.png', '_prediction.png'))
            results[0].save(save_path)
        
        return interpreted_results

    def predict_frame(self, frame_array, confidence_threshold=0.5):
        """Creates an image recognition prediction for a single frame.

        Args:
            frame_array (np.array): Numpy array of the frame.
            confidence_threshold (float, optional): Confidence threshold for detected objects. Defaults to 0.5.
            
        Returns:
            list: List of detected objects. Each object is a tuple containing the object name, bounding box, and confidence.
        """
        results = self.model(frame_array)
        return self._interpret_results(results, confidence_threshold)
        
    def predict_folder(self, folder_path, confidence_threshold=0.5, save_result=False, save_folder='./predictions'):
        """Creates image recognition predictions for all images in a folder.
           If a save folder is provided, the results will be saved with '_prediction' appended.

        Args:
            folder_path (str): Path to the folder containing images.
            confidence_threshold (float, optional): Confidence threshold for detected objects. Defaults to 0.5.
            save_result (bool, optional): Saves the resulting classifications to files. Defaults to False.
            save_folder (str, optional): Folder path to save the results. Defaults to './predictions'.
        
        Returns:
            dict: Dictionary of image paths and their corresponding lists of detected objects.
        """
        img_paths = glob.glob(os.path.join(folder_path, '*.png'))
        if save_result and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        results = {}
        for img_path in img_paths:
            prediction = self.predict_one_image(img_path, save_result=save_result, save_folder=save_folder, confidence_threshold=confidence_threshold)
            results[img_path] = prediction
        
        return results
    
    def _interpret_results(self, results, confidence_threshold=0.5):
        """Interprets the results of the image recognition model with a confidence threshold.

        Args:
            results (list): List of results from the image recognition model.
            confidence_threshold (float, optional): Confidence threshold for detected objects. Defaults to 0.5.

        Returns:
            list: List of detected objects. Each object is a tuple containing the object name, bounding box, and confidence.
        """
        detected_objects = []
        for box, cls in zip(results[0].boxes, results[0].boxes.cls):
            confidence = box.conf.item()  # Access confidence score
            if confidence >= confidence_threshold:
                object_name = self.model.names[int(cls)]
                bounding_box = box.xyxy  # Get the bounding box coordinates
                detected_objects.append((object_name, bounding_box, confidence))
        return detected_objects
    
