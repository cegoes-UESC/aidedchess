from ultralytics.data.converter import convert_coco

PATH = "09-01-24/"


def convert():
    convert_coco(
        "annotations/" + PATH,
        "conversion/converted/",
        use_keypoints=True,
        cls91to80=False,
    )
