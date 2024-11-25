import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib
import time
from stereovision.calibration import StereoCalibration

# Backend Compatibility for Matplotlib
try:
    import tkinter
    import PIL
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt backend
    except ImportError:
        try:
            matplotlib.use('GTK3Agg')  # Try GTK backend
        except ImportError:
            matplotlib.use('Agg')  # Fallback to Agg backend
            print("Warning: Using Agg backend. GUI interaction might be limited.")
print(f"Using Matplotlib backend: {matplotlib.get_backend()}")

# Flag to load settings
loading_settings = 0  # 0 is not currently loading settings.

# Load camera configs
config_path = "cam_config.json"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file {config_path} not found.")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Global variables preset
imageToDisp = './photo.png'
scale_ratio = float(config['scale_ratio'])
image_width = int(int(config['image_width']) * scale_ratio)
image_height = int(int(config['image_height']) * scale_ratio)
photo_height = image_height
photo_width = image_width * 2
image_size = (image_width, image_height)

if not os.path.isfile(imageToDisp):
    print(f'Cannot read image from file \"{imageToDisp}\"')
    exit(0)

pair_img = cv2.imread(imageToDisp, 0)
# Read image and split it into a stereo pair
print('Read and split image...')
imgLeft = pair_img[0:photo_height, 0:image_width]  # Left image
imgRight = pair_img[0:photo_height, image_width:photo_width]  # Right image

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')
rectified_pair = calibration.rectify((imgLeft, imgRight))

# Depth map function with StereoSGBM
SWS = 5
PFS = 5
PFC = 29
MDS = -25
NOD = 128
TTH = 100
UR = 10
SR = 15
SPWS = 100

last_update_time = time.time()

def stereo_depth_map(rectified_pair):
    print(f'SWS={SWS} PFS={PFS} PFC={PFC} MDS={MDS} NOD={NOD} TTH={TTH}')
    print(f'UR={UR} SR={SR} SPWS={SPWS}')
    c, r = rectified_pair[0].shape
    disparity = np.zeros((c, r), np.uint8)
    sbm = cv2.StereoSGBM_create(
        minDisparity=MDS,
        numDisparities=NOD,
        blockSize=SWS,
        P1=8 * 3 * SWS**2,
        P2=32 * 3 * SWS**2,
        disp12MaxDiff=1,
        uniquenessRatio=UR,
        speckleWindowSize=SPWS,
        speckleRange=SR
    )
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
    local_max = disparity.max()
    local_min = disparity.min()
    if local_max != local_min:
        disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))
    else:
        disparity_visual = np.zeros(disparity.shape, dtype=np.uint8)
        print("WARNING: local max and min are the same. Check disparity map calculations.")
    print(f"MAX {local_max}")
    print(f"MIN {local_min}")
    return disparity_visual

disparity = stereo_depth_map(rectified_pair)

# Set up and draw interface
axcolor = 'lightgoldenrodyellow'
fig, _ = plt.subplots(1, 2)
plt.subplots_adjust(left=0.15, bottom=0.5)
plt.subplot(1, 2, 1)
dmObject = plt.imshow(rectified_pair[0], 'gray')

saveax = plt.axes([0.3, 0.38, 0.15, 0.04])
buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')

def save_map_settings(event):
    buttons.label.set_text("Saving...")
    print('Saving to file...')
    result = json.dumps({'SADWindowSize': SWS, 'preFilterSize': PFS, 'preFilterCap': PFC,
                         'minDisparity': MDS, 'numberOfDisparities': NOD, 'textureThreshold': TTH,
                         'uniquenessRatio': UR, 'speckleRange': SR, 'speckleWindowSize': SPWS},
                        sort_keys=True, indent=4, separators=(',', ':'))
    fName = '3dmap_set.txt'
    with open(fName, 'w') as f:
        f.write(result)
    buttons.label.set_text("Save to file")
    print(f'Settings saved to file {fName}')

buttons.on_clicked(save_map_settings)

loadax = plt.axes([0.5, 0.38, 0.15, 0.04])
buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')

def load_map_settings(event):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    loading_settings = 1
    fName = '3dmap_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text("Loading...")
    with open(fName, 'r') as f:
        data = json.load(f)
    sSWS.set_val(data['SADWindowSize'])
    sPFS.set_val(data['preFilterSize'])
    sPFC.set_val(data['preFilterCap'])
    sMDS.set_val(data['minDisparity'])
    sNOD.set_val(data['numberOfDisparities'])
    sTTH.set_val(data['textureThreshold'])
    sUR.set_val(data['uniquenessRatio'])
    sSR.set_val(data['speckleRange'])
    sSPWS.set_val(data['speckleWindowSize'])
    buttonl.label.set_text("Load settings")
    print(f'Parameters loaded from file {fName}')
    print('Redrawing depth map with loaded parameters...')
    loading_settings = 0
    update(0)
    print('Done!')

buttonl.on_clicked(load_map_settings)

plt.subplot(1, 2, 2)
dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')

# Draw interface for adjusting parameters
print('Start interface creation (it takes up to 30 seconds)...')

SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)
PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)
PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)
MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)
NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)
TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)
URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)
SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor)
SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor)

sSWS = Slider(SWSaxe, 'SWS', 5.0, 255.0, valinit=5)
sPFS = Slider(PFSaxe, 'PFS', 5.0, 255.0, valinit=5)
sPFC = Slider(PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=29)
sMDS = Slider(MDSaxe, 'MinDISP', -100.0, 100.0, valinit=-25)
sNOD = Slider(NODaxe, 'NumOfDisp', 16.0, 256.0, valinit=128)
sTTH = Slider(TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=100)
sUR = Slider(URaxe, 'UnicRatio', 1.0, 20.0, valinit=10)
sSR = Slider(SRaxe, 'SpcklRng', 0.0, 40.0, valinit=15)
sSPWS = Slider(SPWSaxe, 'SpklWinSze', 0.0, 300.0, valinit=100)

# Update depth map parameters and redraw
def update(val):
    global last_update_time, SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    if time.time() - last_update_time > 0.2:  # Limit update frequency
        last_update_time = time.time()
        SWS = int(sSWS.val / 2) * 2 + 1  # convert to ODD
        PFS = int(sPFS.val / 2) * 2 + 1
        PFC = int(sPFC.val / 2) * 2 + 1
        MDS = int(sMDS.val)
        NOD = int(sNOD.val / 16) * 16
        TTH = int(sTTH.val)
        UR = int(sUR.val)
        SR = int(sSR.val)
        SPWS = int(sSPWS.val)
        if loading_settings == 0:
            print('Rebuilding depth map')
            disparity = stereo_depth_map(rectified_pair)
            dmObject.set_data(disparity)
            print('Redraw depth map')
            plt.draw()

# Connect update actions to control elements
sSWS.on_changed(update)
sPFS.on_changed(update)
sPFC.on_changed(update)
sMDS.on_changed(update)
sNOD.on_changed(update)
sTTH.on_changed(update)
sUR.on_changed(update)
sSR.on_changed(update)
sSPWS.on_changed(update)

# Save the disparity map button
save_disparity_ax = plt.axes([0.7, 0.38, 0.15, 0.04])
save_disparity_button = Button(save_disparity_ax, 'Save Disparity', color=axcolor, hovercolor='0.975')

def save_disparity_map(event):
    filename = f'disparity_map_{int(time.time())}.png'
    cv2.imwrite(filename, disparity)
    print(f'Saved disparity map as: {filename}')

save_disparity_button.on_clicked(save_disparity_map)

print('Show interface to user')
plt.show()
