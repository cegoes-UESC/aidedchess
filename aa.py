from ultralytics import YOLO
from cvat_sdk import models

import cvat_sdk.auto_annotation as cvataa
from cvat_sdk import make_client
import PIL


class AnnotShapes:
    def __init__(self) -> None:
        self.model = YOLO("best-pose-07-12.pt")

    @property
    def spec() -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(labels=[cvataa.label_spec("board", 1)])

    def detect(self, context, image: PIL.Image.Image):
        results = self.model(image)

        return [
            cvataa.skeleton(label.item(), [x.item() for x in skeleton])
            for result in results
            for skeleton, label in zip(result["keypoints"], result["labels"])
        ]




with make_client(host="localhost", credentials=("user", "password")) as client:
    # annotate task 12345 using Faster R-CNN
    cvataa.annotate_task(client, 41617, AnnotShapes)
