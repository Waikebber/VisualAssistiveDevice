import os, glob
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

class ImgRec:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def predict_one_image(self, img_path, save_result=False, save_folder='./predictions', show_result=False, confidence_threshold=0.5):
        """Creates an image recognition prediction for a single image.
           If a save folder is provided, the result will be saved with '_prediction' appended.

        Args:
            img_path (str): Path to the image.
            save_result (bool, optional): Saves the resulting classification to a file. Defaults to False.
            save_folder (str, optional): Folder path to save the result. Defaults to './predictions'.
            show_result (bool, optional): Displays the resulting image with bounding boxes. Defaults to False.
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
        
        if show_result:
            self._show_results(img, img_path, interpreted_results)

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
        
    def predict_folder(self, folder_path, confidence_threshold=0.5, save_result=False, save_folder='./predictions', show_result=False):
        """Creates image recognition predictions for all images in a folder.
           If a save folder is provided, the results will be saved with '_prediction' appended.

        Args:
            folder_path (str): Path to the folder containing images.
            confidence_threshold (float, optional): Confidence threshold for detected objects. Defaults to 0.5.
            save_result (bool, optional): Saves the resulting classifications to files. Defaults to False.
            save_folder (str, optional): Folder path to save the results. Defaults to './predictions'.
            show_result (bool, optional): Displays the resulting image with bounding boxes. Defaults to False.
        
        Returns:
            dict: Dictionary of image paths and their corresponding lists of detected objects.
        """
        img_paths = glob.glob(os.path.join(folder_path, '*.png'))
        if save_result and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        results = {}
        for img_path in img_paths:
            prediction = self.predict_one_image(img_path, save_result=save_result, save_folder=save_folder, show_result=show_result, confidence_threshold=confidence_threshold)
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
    
    def _show_results(self, image, path, results):
        """Displays the image with bounding boxes, detection names, and confidence scores, closing on key press.

        Args:
            image (PIL.Image): Image to display.
            path (str): Path to the image.
            results (list): List of detected objects. Each object is a tuple containing the object name, bounding box, and confidence.
        """
        # Convert PIL image to a format compatible with OpenCV
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Loop through the results to draw bounding boxes and labels
        for obj_name, bounding_box, confidence in results:
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, bounding_box)
            
            # Draw the bounding box
            cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Prepare label with object name and confidence
            label = f"{obj_name} ({confidence:.2f})"
            
            # Set font and calculate position for label
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_size, _ = cv2.getTextSize(label, font, 0.5, 1)
            label_x, label_y = x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10
            
            # Draw a filled rectangle for the label background for better readability
            cv2.rectangle(img_cv, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y + label_size[1] - 5), (0, 255, 0), cv2.FILLED)
            
            # Put label on the image
            cv2.putText(img_cv, label, (label_x, label_y), font, 0.5, (0, 0, 0), 1)

        # Display the image with bounding boxes in OpenCV
        cv2.imshow(f'Image: {os.path.basename(path)}', img_cv)
        
        # Wait for any key press to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
