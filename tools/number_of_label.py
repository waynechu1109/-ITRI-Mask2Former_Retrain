import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# label_dir = "datasets/merged_dataset/annotations/training"
# label_dir = "/home/lidar/Mask2Former/datasets/mapillary_vistas/training/instances"
label_dir = "/home/lidar/Mask2Former/datasets/mapillary_vistas/training/labels"
max_id = -1
over_ids = set()

for fname in tqdm(os.listdir(label_dir), desc="Scanning annotation images..."):
    if not fname.endswith(".png"):
        continue
    path = os.path.join(label_dir, fname)
    label = np.array(Image.open(path))
    label = label[label != 255]  # skip ignore_label
    if label.size == 0:
        continue
 
    print(f'{path} label: {label}')
    break
