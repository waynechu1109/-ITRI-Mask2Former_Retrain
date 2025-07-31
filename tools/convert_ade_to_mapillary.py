import os
import re
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
 
def load_ade_id2name(objectinfo_path):
    ade_id2name = {}
    with open(objectinfo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Idx") or line.startswith("#"):
                continue
            parts = re.split(r"\t+", line)
            if len(parts) < 5:
                continue
            idx = int(parts[0]) - 1  # ADE ID starts from 1, we want it to start from 0
            raw_label = parts[4].split("//")[0].strip()
            match = re.match(r'^(.*?)(\s+"(.*?)")?$', raw_label)
            if not match:
                continue
            ade_name = match.group(1).strip()
            ade_id2name[idx] = ade_name
    return ade_id2name

def convert_label_img(input_path, output_path, ade_label_map, ade2mapillary, mapillary_name2id):
    label = np.array(Image.open(input_path))
    new_label = np.full_like(label, 255)

    for ade_id, ade_name in ade_label_map.items():
        mapillary_name = ade2mapillary.get(ade_name)
        if mapillary_name is None:
            continue
        mapillary_id = mapillary_name2id[mapillary_name]
        new_label[label == ade_id] = mapillary_id

    Image.fromarray(new_label).save(output_path)

def main(args):
    # Load mappings
    ade_id2name = load_ade_id2name(args.objectinfo)

    print("Load ADE to Mapillary lookup table ...")
    with open(args.ade2mapillary, 'r', encoding='utf-8') as f:
        ade2mapillary = json.load(f)

    mapillary_names = sorted(set(ade2mapillary.values()))
    mapillary_name2id = {name: idx for idx, name in enumerate(mapillary_names)}

    # Save mapillary name to id relationship
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "mapillary_name2id.json"), "w", encoding="utf-8") as f:
        json.dump(mapillary_name2id, f, indent=2, ensure_ascii=False)

    # Start converting
    os.makedirs(args.output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(args.input_dir)):
        if not fname.endswith(".png"):
            continue
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        convert_label_img(in_path, out_path, ade_id2name, ade2mapillary, mapillary_name2id)

    print("Finish conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ADE20K labels to Mapillary-style classes")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to ADE label PNGs (e.g. ADE20K/annotations/train)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save converted label PNGs")
    parser.add_argument("--ade2mapillary", type=str, default="ade2mapillary_name_all.json", help="Mapping from ADE class name to Mapillary class name")
    parser.add_argument("--objectinfo", type=str, default="objectInfo150.txt", help="ADE objectInfo150.txt path")

    args = parser.parse_args()
    main(args)



'''
Usage:

python3 convert_ade_to_mapillary.py \
  --input-dir ade20k/annotations/training \
  --output-dir merged_dataset/annotations/training \
  --ade2mapillary ade2mapillary_name_all.json \
  --objectinfo ade20k/objectInfo150.txt


'''