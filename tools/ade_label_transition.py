import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

input_dir = "datasets/ade20k/annotations/validation"
output_dir = "datasets/ade20k/annotations_modified/validation"
mapping_json = "datasets/ade20k/ade20k_to_mapillary_id.json"  # mapping

os.makedirs(output_dir, exist_ok=True)

# read ADE20k to Mapillary mapping
with open(mapping_json, "r") as f:
    mapping = json.load(f)

# convert "key" to "int" (originally string)
ade2mapillary = {int(k): v for k, v in mapping.items()}

for fname in tqdm(os.listdir(input_dir), desc="ADE20k to Mapillary annotation conversion"):
    if not fname.endswith(".png"):
        continue

    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    ade_label = np.array(Image.open(input_path))

    # default set ignore_label（255）
    mapped_label = np.full_like(ade_label, fill_value=255)

    for ade_id, mapillary_id in ade2mapillary.items():
        mapped_label[ade_label == ade_id] = mapillary_id

    Image.fromarray(mapped_label.astype(np.uint8)).save(output_path)

print(f"Complete all the conversion, output: {output_dir}")
