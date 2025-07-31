import os
import random
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm

image_folder = "../datasets/merged_dataset/images/training"
annotation_folder = "../datasets/merged_dataset/annotations/training"
filename_list_path = "../datasets/merged_dataset/training_images_without_stair.txt"
 
aug_image_output = "../datasets/merged_dataset/images/training"
aug_annotation_output = "../datasets/merged_dataset/annotations/training"

# os.makedirs(aug_image_output, exist_ok=True)
# os.makedirs(aug_annotation_output, exist_ok=True)

with open(filename_list_path, 'r') as f:
    filenames = [line.strip() for line in f.readlines()]

def random_augment(image, annotation):
    # # horizontally flip
    # # if random.random() < 0.5:
    # image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # annotation = annotation.transpose(Image.FLIP_LEFT_RIGHT)

    # # random brightness adjustment
    # if random.random() < 0.5:
    #     enhancer = ImageEnhance.Brightness(image)
    #     image = enhancer.enhance(random.uniform(0.7, 1.3))

    # # random rotation
    # if random.random() < 0.5:
    #     angle = random.uniform(-10, 10)
    #     image = image.rotate(angle, resample=Image.BILINEAR)
    #     annotation = annotation.rotate(angle, resample=Image.NEAREST)
    
    # downsample
    new_size = (683, 512)
    image = image.resize(new_size, resample=Image.BILINEAR)
    annotation = annotation.resize(new_size, resample=Image.NEAREST)

    return image, annotation

for filename in tqdm(filenames, desc="Augmenting data"):
    image_path = os.path.join(image_folder, filename.replace(".png", ".jpg"))
    annotation_path = os.path.join(annotation_folder, filename)

    # if not os.path.exists(image_path) or not os.path.exists(annotation_path):
    #     continue

    image = Image.open(image_path).convert("RGB")
    annotation = Image.open(annotation_path)

    # augmentation
    aug_img, aug_ann = random_augment(image, annotation)

    base_name = os.path.splitext(filename)[0]
    
    # print(f'Start saving...')
    aug_img.save(os.path.join(aug_image_output, base_name + ".jpg"))
    aug_ann.save(os.path.join(aug_annotation_output, base_name + ".png"))
