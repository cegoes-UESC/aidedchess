import math
import os
import random
from pathlib import Path


def split_datasets():
    PATH = "datasets/chess"

    labels_dir = Path(PATH + "/labels")
    images_dir = Path(PATH + "/images")

    labels = list(labels_dir.glob("*.txt"))

    random.shuffle(labels)

    IMAGE_COUNT = len(labels)
    RATIO = 0.8

    train_count = math.ceil(IMAGE_COUNT * RATIO)
    val_count = IMAGE_COUNT - train_count

    print(IMAGE_COUNT, RATIO, train_count, val_count, train_count + val_count)

    val_images = labels[:val_count]
    train_images = labels[val_count:]

    val_path = Path(PATH + "/labels/val")
    train_path = Path(PATH + "/labels/train")

    images_val_path = Path(PATH + "/images/val")
    images_train_path = Path(PATH + "/images/train")

    print("Copying VAL labels")
    for i in val_images:
        name = i.name.split(".")[0]
        imagename = name + ".JPG"
        os.rename(i.resolve(), (val_path / i.name).resolve())
        os.rename(
            (images_dir / imagename).resolve(),
            (images_val_path / imagename).resolve(),
        )

    print("Copying TRAIN labels")
    for i in train_images:
        name = i.name.split(".")[0]
        imagename = name + ".JPG"
        os.rename(i.resolve(), (train_path / i.name).resolve())
        os.rename(
            (images_dir / imagename).resolve(),
            (images_train_path / imagename).resolve(),
        )
