import json
import cv2

def load_map_settings_with_sbm(file_path):
    """
    Load settings from a file and initialize the StereoBM object.

    Args:
        file_path (str): Path to the file containing the stereo block matching settings.

    Returns:
        cv2.StereoBM: Configured StereoBM object.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']

        # Initialize and configure StereoBM object
        sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(PFS)
        sbm.setPreFilterCap(PFC)
        sbm.setMinDisparity(MDS)
        sbm.setTextureThreshold(TTH)
        sbm.setUniquenessRatio(UR)
        sbm.setSpeckleRange(SR)
        sbm.setSpeckleWindowSize(SPWS)

        print(f"Parameters loaded from file {file_path}")

        return sbm

    except Exception as e:
        print(f"Error loading settings from file {file_path}: {e}")
        raise
