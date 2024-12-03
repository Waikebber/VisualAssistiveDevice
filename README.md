# Visual Assistive Device

A Raspberry Pi-based visual assistance system that combines stereo vision for depth perception with real-time object detection and audio feedback. This system is designed to be mounted on a hat for hands-free operation.

## Features

- **Stereo Vision**: Uses dual cameras to create depth maps and measure distances
- **Object Detection**: Real-time object recognition using YOLOv8
- **Audio Feedback**: Text-to-speech notifications for:
  - Distance warnings when objects are too close
  - Detected objects and their distances upon button press
- **Interactive Control**: Single-button interface for on-demand object detection
- **Visual Display**: Optional real-time display of stereo vision and depth mapping (can be disabled for standalone operation)

## Hardware Requirements

- Raspberry Pi
  - [CM4](https://www.raspberrypi.com/products/compute-module-4/?variant=raspberry-pi-cm4001000)
- Stereo camera module
  - [Dual IMX219](https://www.waveshare.com/imx219-83-stereo-camera.htm)
- Push button for interaction
- Speaker/headphone for audio feedback
- Power supply

## Software Dependencies
Please use the bash script as follows for the environemnt:

```bash
chmod +x setup_env.sh
./setup_env.sh
source yolo-env/bin/activate
```

## Project Structure

```
project_root/
├── main.py                   # Basic stereo vision script
├── main-ir.py                # Stereo vision with image recognition
├── image_rec/              # Image recognition package
│   ├── __init__.py
│   ├── img_rec.py           # YOLO implementation
│   └── stereoImgRec.py      # Stereo image recognition helpers
├── speakers/               # Audio feedback package
│   ├── __init__.py
│   └── generative_audio.py   # Text-to-speech function
└── stereo_calibration/     # Calibration files
    ├── cam_config.json       # Camera configuration
    ├── 3dmap_set.txt        # Stereo matching parameters
    ├── tuning_helper.py      # Helper for loading 3dmap_set.txt
    ├── rectify.py           # Image rectification and depth mapping
    └── calib_result/         # Camera calibration data
```

## Configuration

1. Camera Setup
   - Modify `stereo_calibration/cam_config.json` for your camera parameters
   - Baseline distance
   - Field of view
   - Resolution settings

2. Distance Thresholds
   - Set warning distance in meters (default: 3.5m)
   - Adjust confidence threshold for object detection (default: 0.6)

## Usage

1. Basic Stereo Vision (without object detection):
```bash
python main.py
```

2. Full System (with object detection):
```bash
python main-ir.py
```

3. Controls:
   - Press button to trigger object detection and distance measurement
   - Press 'q' to quit (when display is enabled)

## Operation Modes

1. **Display Mode**:
   - Shows real-time camera feeds and depth map
   - Useful for testing and calibration

2. **Headless Mode**:
   - Runs without visual output
   - Suitable for standalone operation
   - Reduces processing overhead

## Audio Feedback

The system provides audio notifications for:
- Proximity warnings when objects are closer than the threshold
- Object detection results including:
  - Object type
  - Distance in meters
  - Location relative to the user

## Calibration

The stereo cameras need to be calibrated before use:
1. Use calibration patterns
2. Save calibration data to `stereo_calibration/calib_result/`
3. Adjust stereo matching parameters in `stereo_calibration/3dmap_set.txt`

## Performance Considerations

- Resolution affects processing speed
- Adjustable parameters:
  - Image scale ratio
  - Detection confidence threshold (default: 0.6)
  - Distance warning threshold (default: 3.5m)
  - Frame processing rate

## Safety Notes

- This is an assistive device and should not be relied upon as the sole means of navigation
- Always maintain awareness of your surroundings
- Regular calibration checks are recommended
- Verify detection accuracy in different lighting conditions

## License

### GNU AFFERO GENERAL PUBLIC LICENSE
Version 3, 19 November 2007

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Acknowledgments and Attributions

#### Stereovision Implementation
- Inspired by [Realizator's stereopi-tutorial repo](https://github.com/realizator/stereopi-tutorial)
- Licensed under GPL-3.0

This project builds upon existing open-source work while maintaining compliance with their respective licenses. The original stereovision implementation's GPL-3.0 license is compatible with our project's AGPL-3.0 license.

For the complete license text, see the LICENSE file in the root directory or visit:
- [GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
- [Original GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
