import json
from pathlib import Path

from ultralytics.data.converter import convert_coco


def handle_files():
    path = Path("annotations/")
    files = list(path.glob("*.json"))

    for f in files:
        file = open(f, "r")
        data = json.load(file)
        file.close()
        for idx, c in enumerate(data["categories"]):
            data["categories"][idx]["id"] = int(c["id"] / 5 + 1)
        for idx, a in enumerate(data["annotations"]):
            data["annotations"][idx]["category_id"] = int(a["category_id"] / 5 + 1)
            bbox = data["annotations"][idx]["bbox"]
            if bbox[2] == 0:
                data["annotations"][idx]["bbox"][2] = 50
            if bbox[3] == 0:
                data["annotations"][idx]["bbox"][3] = 50
            keypoints = data["annotations"][idx]["keypoints"]
            v1, v2, v3, v4 = keypoints[2], keypoints[5], keypoints[8], keypoints[11]
            for idx, v in enumerate([v1, v2, v3, v4]):
                if v == 0:
                    data["annotations"][idx]["keypoints"][idx - 1] = 0
                    data["annotations"][idx]["keypoints"][idx - 2] = 0
        file = open(f, "w")
        json.dump(data, file)
        file.close()


def convert():
    convert_coco(
        "annotations/",
        "conversion/converted/",
        use_keypoints=True,
        cls91to80=False,
    )
    converted = Path("conversion/converted/labels")
    labels = converted.rglob("*.txt")
    for file in labels:
        file.rename("datasets/chess/labels/" + file.name)


def run():
    handle_files()
    convert()
