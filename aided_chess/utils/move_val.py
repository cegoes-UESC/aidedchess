import json
import os
from sys import argv

p = argv[1]


file = open("annotations/" + p + "/keypoints.json", "r")

data = json.load(file)

ids = data["val"]

for name in ids:
    os.rename(
        "datasets/albumented/images/train/" + name,
        "datasets/albumented/images/val/" + name,
    )
    n = name.split(".")[0]
    os.rename(
        "datasets/albumented/labels/train/" + n + ".txt",
        "datasets/albumented/labels/val/" + n + ".txt",
    )
