import json
import cv2

import json
import cv2

def load_map_settings_with_sgbm(file_path):
    """
    Load settings from a file and initialize the StereoSGBM object.

    Args:
        file_path (str): Path to the file containing the stereo block matching settings.

    Returns:
        cv2.StereoSGBM: Configured StereoSGBM object.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract stereo matching settings from JSON
        SWS = data['SADWindowSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']

        # Initialize and configure StereoSGBM object
        sgbm = cv2.StereoSGBM_create(
            minDisparity=MDS,
            numDisparities=NOD,
            blockSize=SWS,
            P1=8 * 3 * SWS ** 2,
            P2=32 * 3 * SWS ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=UR,
            speckleWindowSize=SPWS,
            speckleRange=SR,
            preFilterCap=PFC  # Include directly during creation
        )

        print(f"Parameters loaded from file {file_path}")

        return sgbm

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file {file_path}.")
        raise
    except KeyError as e:
        print(f"Error: Missing key in JSON settings: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        raise


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

        # Extract stereo matching settings from JSON
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
        sbm.setPreFilterSize(PFS)
        sbm.setPreFilterCap(PFC)
        sbm.setMinDisparity(MDS)
        sbm.setTextureThreshold(TTH)
        sbm.setUniquenessRatio(UR)
        sbm.setSpeckleRange(SR)
        sbm.setSpeckleWindowSize(SPWS)

        print(f"Parameters loaded from file {file_path}")

        return sbm

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file {file_path}.")
        raise
    except KeyError as e:
        print(f"Error: Missing key in JSON settings: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        raise
