import numpy as np
from PIL import Image
import sys

image_path_1 = "/home/lidar/tao_tutorials/notebooks/tao_launcher_starter_kit/mask2former/datasets/coco/panoptic_train2017/000000000009.png"  # ← 修改成你的圖檔路徑
image_path_2 = "/home/lidar/Mask2Former/datasets/coco/panoptic_semseg_train2017/000000000009.png"
label_img_1 = np.array(Image.open(image_path_1))
label_img_2 = np.array(Image.open(image_path_2))

unique_labels_1 = np.unique(label_img_1)
unique_labels_2 = np.unique(label_img_2)

print(f"Img 1: {image_path_1}")
print(f"Find {len(unique_labels_1)} unique label: ")
print(sorted(unique_labels_1.tolist()))

print("\nThe number pixels of every label in Img1: ")
unique, counts = np.unique(label_img_1, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label {u}: {c} pixels")

print('\n\n')

print(f"Img 2: {image_path_2}")
print(f"Find {len(unique_labels_2)} unique label: ")
print(sorted(unique_labels_2.tolist()))

print("\nThe number pixels of every label in Img2: ")
unique, counts = np.unique(label_img_2, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label {u}: {c} pixels")
