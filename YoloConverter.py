import cv2 as cv
from pathlib import Path
import os, json

BASE_DIR = Path("datasets/albumented")

LABELS_PATH = BASE_DIR / "labels" / "train"
IMAGES_PATH = BASE_DIR / "images" / "train"

labels = os.listdir(str(LABELS_PATH.absolute()))
images = os.listdir(str(IMAGES_PATH.absolute()))

images = list(map(lambda i: i.split("."), images))

data = {
    "images": [],
    "annotations": [],
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": "",
    },
    "categories": [
        {
            "id": 1,
            "name": "board",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 6,
            "name": "black-pawn",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 11,
            "name": "white-pawn",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 16,
            "name": "black-rook",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 21,
            "name": "white-rook",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 26,
            "name": "black-bishop",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 31,
            "name": "white-bishop",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 36,
            "name": "black-knight",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 41,
            "name": "white-knight",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 46,
            "name": "black-queen",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 51,
            "name": "white-queen",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 56,
            "name": "black-king",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
        {
            "id": 61,
            "name": "white-king",
            "supercategory": "",
            "keypoints": ["1", "2", "3", "4"],
            "skeleton": [[2, 3], [1, 2], [4, 1], [3, 4]],
        },
    ],
}

image_id = 1
ann_id = 1


def add_annotations(id, label_name, h, w, ann_id):
    path = LABELS_PATH / label_name
    file = open(path, "r")
    for line in file:
        line = line.strip()
        d = line.split(" ")
        class_id = int(d[0]) * 5 + 1
        bbox = d[1:5]
        bbox[0] = float(bbox[0]) * w
        bbox[1] = float(bbox[1]) * h
        bbox[2] = float(bbox[2]) * w
        bbox[3] = float(bbox[3]) * h
        keypoints = d[5:]
        num_ks = 0
        for idx, k in enumerate(keypoints):
            if idx in [2, 5, 8, 11]:
                keypoints[idx] = int(keypoints[idx])
                if keypoints[idx] != 0:
                    num_ks = num_ks + 1
                continue
            if idx in [0, 3, 6, 9]:
                keypoints[idx] = float(keypoints[idx]) * w
            else:
                keypoints[idx] = float(keypoints[idx]) * h
        data["annotations"].append(
            {
                "id": ann_id,
                "image_id": id,
                "category_id": class_id,
                "segmentation": [],
                "area": bbox[2] * bbox[3],
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {"occluded": False},
                "keypoints": keypoints,
                "num_keypoints": num_ks,
            }
        )
        ann_id = ann_id + 1
    return ann_id


for label in labels:

    name = label.split(".")[0]
    image = list(filter(lambda i: i[0] == name, images))[0]

    image_name = ".".join(image)
    image_path = IMAGES_PATH / image_name

    image = cv.imread(str(image_path.absolute()), cv.IMREAD_COLOR)
    h, w = image.shape[:2]

    data["images"].append(
        {
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": image_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        },
    )
    ann_id = add_annotations(image_id, label, h, w, ann_id)
    image_id = image_id + 1
    print(f'Image "{image_name}" done!')

out_path = Path("annotations")

out = open(str((out_path / "yolo-to-coco.json").absolute()), "w")

json.dump(data, out)
