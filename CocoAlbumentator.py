import cv2 as cv
from pathlib import Path
import albumentations as A
from sys import argv
import random, math, json

VAL_PROP = 0.2

path = argv[1]

file_path = "annotations/" + path + "/keypoints.json"

file = open(file_path, "r")

data = json.load(file)
file.close()

images = list.copy(data["images"])

GET = math.ceil(len(images) * VAL_PROP)

random.shuffle(images)
val_ids = list(map(lambda x: x["id"], images[:GET]))

data["val"] = val_ids


file = open(file_path, "w")
json.dump(data, file)
file.close()

for im in data["images"]:
    pass


def get_image_annotations(image_id):
    pass


def add_image(id, name, w, h):
    pass


def add_annotation(id, image_id, category_id, bboxes, keypoints):
    pass
