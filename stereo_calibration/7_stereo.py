# %%
from rectify import rectify_imgs, make_disparity_map
import cv2
from random import random
from matplotlib import pyplot as plt

# %%
INPUT_DIR = '../data/stereo_images/TEST'
CALIBRATION_DIR = '../data/stereo_images/scenes/calibration_results'
BASELINE = 0.06
DISTANCE_FACTOR = BASELINE * 5/8 # (25 / 39)

# %%
file_num = int(random() * 10) + 1
selected_file = f'{INPUT_DIR}/raw/{file_num}.png'
print(f'Processing {selected_file}')
image = cv2.imread(selected_file)

plt.imshow(image)
plt.axis('off')  # Hide axes
plt.gcf().set_size_inches(10, 10)  # Set the figure size to make the image larger
plt.show()

# %%
left = cv2.imread(f'{INPUT_DIR}/left/{file_num}.png')
right = cv2.imread(f'{INPUT_DIR}/right/{file_num}.png')

# %%
if left is not None and right is not None:
	left_rectified, right_rectified, Q, focal_length = rectify_imgs(left, right, CALIBRATION_DIR)
else:
	print("Error: One or both of the images are not loaded correctly.")

# %%
min_disp = 0
num_disp = 16 * 2
block_size = 10
disparity_map = make_disparity_map(left_rectified, right_rectified, min_disp, num_disp, block_size)

# Display disparity map
plt.imshow(disparity_map, 'gray')
plt.title("Disparity Map")
plt.colorbar()
plt.show()


# %%

# Step 4: Convert Disparity Map to Depth Map
depth_map = cv2.reprojectImageTo3D(disparity_map, Q)

# Function to handle mouse click event
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # Get the coordinates where the user clicked
        x, y = int(event.xdata), int(event.ydata)

        # Ensure the clicked point is within bounds of the image
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            # Check if disparity value is valid
            if disparity_map[y, x] > min_disp:
                # Get depth (z-coordinate)
                distance = depth_map[y, x, 2] * DISTANCE_FACTOR
                print(f"Distance at point ({x}, {y}): {distance:.2f} meters")
            else:
                print(f"No valid disparity at point ({x}, {y})")
        else:
            print(f"Clicked point ({x}, {y}) is out of bounds.")

# Create a figure with two subplots to display the rectified left image and depth map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Display the rectified left image on the left
ax1.imshow(cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
ax1.set_title("Rectified Left Image")
ax1.axis('off')  # Hide axis labels

# Display the depth map on the right
depth_display = ax2.imshow(depth_map[:, :, 2], 'jet')  # Z values represent the depth
ax2.set_title("Depth Map (in meters)")
plt.colorbar(depth_display, ax=ax2)

# Connect the click event to the handler for the depth map only
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
