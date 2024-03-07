import cv2 as cv
from pathlib import Path
import albumentations as A
from sys import argv
import random, math, json
from time import time_ns

VAL_PROP = 0.2

path = argv[1]

file_path = "annotations/" + path + "/keypoints.json"

file = open(file_path, "r")

data = json.load(file)
file.close()

images = list.copy(data["images"])

GET = math.ceil(len(images) * VAL_PROP)

random.shuffle(images)
val_ids = list(map(lambda x: x["file_name"], images[:GET]))

data["val"] = val_ids


file = open("annotations/" + path + "/out_keypoints.json", "w")
json.dump(data, file)
file.close()


def get_image_annotations(image_id):
    return list(filter(lambda x: x["image_id"] == image_id, data["annotations"]))


def add_image(id, name, w, h):
    pass


def add_annotation(id, image_id, category_id, bboxes, keypoints):
    ann = {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bboxes": bboxes,
        "keypoints": keypoints,
        "num_keypoints": 4,
        "attributes": {"occluded": False},
        "segmentation": [],
        "area": bboxes[2] * bboxes[3],
        "iscrowd": 0,
    }


current_image_id = data["images"][len(data["images"]) - 1]
current_ann_id = data["annotations"][len(data["annotations"]) - 1]

for im in data["images"]:
    id = im["id"]
    filename = im["file_name"]
    p = str(Path("datasets/albumented/images/train/" + filename).absolute())
    image = cv.imread(p, cv.IMREAD_COLOR)
    out_name = filename + "_" + str(time_ns())

    h, w = image.shape[:2]
    add_image(current_image_id, out_name + ".jpg", w, h)
    ann = get_image_annotations(id)
