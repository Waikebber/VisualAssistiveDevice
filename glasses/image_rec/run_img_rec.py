import os
import sys
from vision.img_rec import ImgRec

def run_image_recognition(input_path, confidence_threshold=0.5, save_result=False):
    """Runs image recognition on a single image or all images in a folder.

    Args:
        input_path (str): Path to an image file or folder containing images.
        confidence_threshold (float, optional): Confidence threshold for detected objects. Defaults to 0.5.
        save_result (bool, optional): Whether to save the resulting images with predictions. Defaults to False.
    """
    img_rec = ImgRec()

    if os.path.isdir(input_path):
        # Run prediction on all images in the folder
        print(f"Running image recognition on all images in folder: {input_path}")
        results = img_rec.predict_folder(input_path, confidence_threshold=confidence_threshold, save_result=save_result)
        
        for img_path, detections in results.items():
            print(f"\nImage: {img_path}")
            for obj_name, bbox, confidence in detections:
                print(f"Detected: {obj_name} with confidence {confidence:.2f} at {bbox}")
    elif os.path.isfile(input_path):
        # Run prediction on a single image
        print(f"Running image recognition on image: {input_path}")
        detections = img_rec.predict_one_image(input_path, confidence_threshold=confidence_threshold, save_result=save_result)
        
        print(f"\nImage: {input_path}")
        for obj_name, bbox, confidence in detections:
            print(f"Detected: {obj_name} with confidence {confidence:.2f} at {bbox}")
    else:
        print("Invalid path. Please provide a valid image file or folder path.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_image_recognition.py <input_path> [confidence_threshold] [save_result]")
        print("Example: python run_image_recognition.py ./images 0.5 True True")
        sys.exit(1)

    input_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    save_result = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    run_image_recognition(input_path, confidence_threshold, save_result)
