import cv2

def initialize_windows():
    """Initialize and position OpenCV windows."""
    cv2.namedWindow("Image")
    cv2.moveWindow("Image", 50, 100)
    cv2.namedWindow("left")
    cv2.moveWindow("left", 450, 100)
    cv2.namedWindow("right")
    cv2.moveWindow("right", 850, 100)

def update_displays(disparity_map, left_img, right_img):
    """Update all display windows."""
    cv2.imshow("Image", disparity_map)
    cv2.imshow("left", left_img)
    cv2.imshow("right", right_img)
    return cv2.waitKey(1) & 0xFF