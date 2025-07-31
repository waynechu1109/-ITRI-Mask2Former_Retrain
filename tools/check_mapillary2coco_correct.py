import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np

with open("/home/lidar/Mask2Former_retrain/datasets/merged_dataset/annotations/massive_annotations/image4953_info.json", "r") as f:
    data = json.load(f)

image_info = data["images"][0]
width, height = image_info["width"], image_info["height"]

# build a blank canvas
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(canvas)
ax.set_title(f"Segmentations for image: {image_info['file_name']}")
ax.axis("off")

colors = plt.cm.get_cmap('tab20', len(data["annotations"]))

# plot all segmentations
for i, ann in enumerate(data["annotations"]):
    color = colors(i)[:3]  # RGB
    segmentation = ann["segmentation"]
    for seg in segmentation:
        poly = np.array(seg).reshape((-1, 2))
        patch = patches.Polygon(poly, closed=True, fill=True, edgecolor=color, facecolor=color, alpha=0.5, linewidth=1)
        ax.add_patch(patch)

plt.tight_layout()
plt.show()
