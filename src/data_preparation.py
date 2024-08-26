import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import cv2
import glob

def downscale_image_by_two(image):
    height, width = image.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("The dimensions of the image must be even.")
    downscaled_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    expanded_image = cv2.resize(downscaled_image, (width, height), interpolation=cv2.INTER_NEAREST)
    return expanded_image

image_paths = glob.glob('../input/train_photos/*')
h5_file_path = '../input/train_data.h5'

original_patches = []
downscaled_patches = []

for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        continue  
    
    for y in range(0, image.shape[0] - 33, 34):
        for x in range(0, image.shape[1] - 33, 34):
            patch = image[y:y+34, x:x+34]
            if patch.shape[0] == 34 and patch.shape[1] == 34:
                downscaled_patch = downscale_image_by_two(patch)
                original_patches.append(patch)
                downscaled_patches.append(downscaled_patch)
                
original_patches = np.array(original_patches)
original_patches = original_patches.astype(np.float32)/255.0
downscaled_patches = np.array(downscaled_patches)
downscaled_patches = downscaled_patches.astype(np.float32)/255.0

with h5py.File(h5_file_path, 'w') as h5_file:
    h5_file.create_dataset('data', data=np.expand_dims(downscaled_patches, axis=1))
    h5_file.create_dataset('label', data=np.expand_dims(original_patches, axis=1))
