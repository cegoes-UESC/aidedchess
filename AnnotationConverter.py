import json
from pathlib import Path
from ultralytics.data.converter import convert_coco

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


convert()
