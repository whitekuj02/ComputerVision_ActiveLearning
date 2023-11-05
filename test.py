import os
import cv2 
'''
list_train = os.listdir("/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/train")
print(list_train)
for i in list_train:
    a = cv2.imread("/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/train/"+i, cv2.IMREAD_COLOR)
    print(a.shape)
'''

import os
import cv2
import random

# Path to the training images directory
train_dir = "/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/train"

# Path where you want to save the blurred image
save_dir = "/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/"
save_filename = "test0_0.jpg"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# List all files in the training directory
list_train = os.listdir(train_dir)

# Select one random image from the list
random_filename = random.choice(list_train)

# Construct the full file path
file_path = os.path.join(train_dir, random_filename)

# Read the image
image = cv2.imread(file_path, cv2.IMREAD_COLOR)

# Check if the image was correctly loaded
if image is not None:
    print(f"Selected image: {random_filename}")
    print(f"Image shape: {image.shape}")
    
    # Apply Gaussian Blur to the image
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    
    # Save the blurred image to the specified path
    save_path = os.path.join(save_dir, save_filename)
    cv2.imwrite(save_path, blurred_image)
    print(f"Blurred image saved as: {save_path}")
else:
    print(f"Error loading image: {random_filename}")
