# Make sure that the following commands have been ran for rasberry pi usage:
#   sudo apt update
#   sudo apt install python3-opencv
#   sudo apt install libopencv-dev

import cv2

left_cam = cv2.VideoCapture(0)  # Left camera
right_cam = cv2.VideoCapture(1)  # Right camera

while True:
    retL, left_frame = left_cam.read()
    retR, right_frame = right_cam.read()

    if retL and retR:
        # Show both frames side by side
        combined = cv2.hconcat([left_frame, right_frame])
        cv2.imshow('Stereo Camera', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
