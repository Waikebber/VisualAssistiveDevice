import os

# Define the directories
base_dir = "../data/stereo_images/dataset1/"
subdirs = ["left", "right", "raw"]

# Get a list of all the image numbers that are present in each subdirectory
image_sets = []
for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    images = [int(img_name.split('.')[0]) for img_name in os.listdir(subdir_path) if img_name.endswith('.png')]
    image_sets.append(set(images))

# Get the union of all image sets to include all image numbers that exist in any of the subdirectories
all_images = set.union(*image_sets)

# Get the intersection of all image sets to only include images that are present in all subdirectories
common_images = set.intersection(*image_sets)

# Determine missing images (images that exist in some but not all subdirectories)
missing_images = all_images - common_images

# Remove images that are missing from any of the subdirectories
for missing_number in missing_images:
    for subdir in subdirs:
        missing_file = os.path.join(base_dir, subdir, f"{missing_number}.png")
        if os.path.isfile(missing_file):
            os.remove(missing_file)

# Sort the remaining common images in ascending order
common_images = sorted(list(common_images))

# Rename the images sequentially without skipping a number
new_image_counter = 1
for old_number in common_images:
    for subdir in subdirs:
        old_name = os.path.join(base_dir, subdir, f"{old_number}.png")
        new_name = os.path.join(base_dir, subdir, f"{new_image_counter}.png")
        os.rename(old_name, new_name)
    new_image_counter += 1

print("Renaming completed.")
