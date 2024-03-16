from ultralytics.data.converter import convert_coco
from sys import argv
from pathlib import Path
import json

PATH = argv[1]

path = Path("annotations/" + PATH)
files = list(path.glob("*.json"))

for f in files:
    file = open(f, "r")
    data = json.load(file)
    file.close()
    for idx, c in enumerate(data["categories"]):
        data["categories"][idx]["id"] = int(c["id"] / 5 + 1)
    for idx, a in enumerate(data["annotations"]):
        data["annotations"][idx]["category_id"] = int(a["category_id"] / 5 + 1)
    file = open(f, "w")
    json.dump(data, file)
    file.close()


def convert():
    convert_coco(
        "annotations/" + PATH,
        "conversion/converted/",
        use_keypoints=True,
        cls91to80=False,
    )


convert()
