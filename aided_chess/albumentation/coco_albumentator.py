import random
import math
import json
from sys import argv
from pathlib import Path
from time import time_ns
import cv2 as cv
import numpy as np
from .albument import augment

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


def get_image_annotations(image_id):
    return list(filter(lambda x: x["image_id"] == image_id, data["annotations"]))


def add_image(id, name, w, h):
    data["images"].append(
        {
            "id": id,
            "width": w,
            "height": h,
            "file_name": name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
    )


def add_annotation(id, image_id, category_id, bboxes, keypoints):
    kpts = keypoints.reshape(12)
    ann = {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bboxes,
        "keypoints": [k for i, k in enumerate(kpts)],
        "num_keypoints": 4,
        "attributes": {"occluded": False},
        "segmentation": [],
        "area": bboxes[2] * bboxes[3],
        "iscrowd": 0,
    }
    data["annotations"].append(ann)


current_image_id = data["images"][len(data["images"]) - 1]["id"]
current_ann_id = data["annotations"][len(data["annotations"]) - 1]["id"]

images = data["images"].copy()

for im in images:
    id = im["id"]
    filename = im["file_name"]
    if filename in val_ids:
        continue

    p = str(Path("datasets/albumented/images/train/" + filename).absolute())
    image = cv.imread(p, cv.IMREAD_COLOR)
    out_name = filename.split(".")[0] + "_" + str(time_ns())

    h, w = image.shape[:2]
    current_image_id = current_image_id + 1
    add_image(current_image_id, out_name + ".jpg", w, h)
    ann = get_image_annotations(id)

    bboxes, keypoints, visibility = [], [], []

    targets, arguments = {}, {}
    size = 0

    for item_idx, a in enumerate(ann):
        size = size + 1
        bboxes = [], keypoints = []

        if item_idx != 0:
            targets["bboxes" + str(item_idx - 1)] = "bboxes"
            targets["keypoints" + str(item_idx - 1)] = "keypoints"

        bboxes.extend(a["bbox"])
        bboxes.extend(a["category_id"])

        kp_np = np.array(a["keypoints"])
        kp_np = kp_np.reshape((4, 3))
        v = []
        for k in kp_np:
            keypoints.append([k[0], k[1]])
            v.append(k[2])
        visibility.append(v)

        if item_idx == 0:
            arguments["bboxes"] = [bboxes]
            arguments["keypoints"] = keypoints
        else:
            arguments["bboxes" + str(item_idx - 1)] = [bboxes]
            arguments["keypoints" + str(item_idx - 1)] = keypoints

    for idx, bbb in enumerate(bboxes):
        if bbb[2] == 0:
            bboxes[idx][2] = 1
        if bbb[3] == 0:
            bboxes[idx][3] = 1

    im, bboxes, kpts, classes = augment(image, targets, arguments, size, True)

    h, w = im.shape[:2]
    kpts = kpts.reshape((len(classes), 4, 3))

    for j, g in enumerate(kpts):
        for m, k in enumerate(g):
            if k[2] != 0:
                print(visibility[j])
                kpts[j][m][2] = visibility[j][m]

    cv.imwrite(
        str(Path("datasets/albumented/images/train/" + out_name + ".jpg").absolute()),
        im * 255,
    )

    for idx, i in enumerate(classes):
        current_ann_id = current_ann_id + 1
        add_annotation(
            current_ann_id, current_image_id, i, bboxes[idx], keypoints=kpts[idx]
        )


file = open("annotations/" + path + "/out_keypoints.json", "w")
json.dump(data, file)
file.close()
