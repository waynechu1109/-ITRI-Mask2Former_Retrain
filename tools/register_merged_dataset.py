import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
 
def register_merged_dataset(name, image_dir, label_dir, class_json_path):
    with open(class_json_path, "r") as f:
        config = json.load(f)

    # make sure config is dict and contains "labels"
    if isinstance(config, dict) and "labels" in config:
        class_list = config["labels"]
    elif isinstance(config, list):
        class_list = config
    else:
        raise ValueError("config_merged.json format error: should be a list or a dict containing the 'labels' field.")

    # retrieve all readables for classes name
    stuff_classes = [item["readable"] for item in class_list]
    # register designated color for inference mask color
    stuff_colors = [l["color"] for l in class_list]

    # register dataset
    DatasetCatalog.register(name, lambda: load_sem_seg(label_dir, image_dir))
    MetadataCatalog.get(name).set(
        stuff_classes=stuff_classes,
        stuff_colors=stuff_colors,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=image_dir,
        sem_seg_root=label_dir,
    )

    print(f"Registration complete: {name}, number of classes: {len(stuff_classes)}")
