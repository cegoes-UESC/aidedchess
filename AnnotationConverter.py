from ultralytics.data.converter import convert_coco
from sys import argv

PATH = argv[1]


def convert():
    convert_coco(
        "annotations/" + PATH,
        "conversion/converted/",
        use_keypoints=True,
        cls91to80=False,
    )


convert()
