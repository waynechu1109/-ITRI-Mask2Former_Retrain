import os
import numpy as np
from PIL import Image
from tqdm import tqdm

annotation_folder = "../datasets/merged_dataset/annotations/training"
target_ids = {150}  # The ID of the stair class is 150

valid_exts = ('.png', '.jpg', '.jpeg')
 
# get all the files' name
all_files = [f for f in os.listdir(annotation_folder) if f.lower().endswith(valid_exts)]

for filename in tqdm(all_files, desc="Processing images"):
    path = os.path.join(annotation_folder, filename)
    img = Image.open(path)
    np_img = np.array(img)

    unique_ids = set(np.unique(np_img))

    if target_ids & unique_ids:
        # outfile.write(filename + '\n')
        print(f'Finding {filename} with id 150')
