import PIL
from typing import List
from cvat_sdk import models
from ultralytics import YOLO
from cvat_sdk import make_client
import cvat_sdk.auto_annotation as cvataa


class AnnotShapes:
    def __init__(self) -> None:
        self.model = YOLO("models/best-pose-07-12.pt")
        self.model.eval()

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.skeleton_label_spec(name, id)
                for (name, id) in self.model.names.items()
            ],
        )

    def detect(
        self, context, image: PIL.Image.Image
    ) -> List[models.LabeledShapeRequest]:
        results = self.model(image)

        return [
            cvataa.skeleton(label.item(), skeleton.xy.tolist())
            for result in results
            for skeleton, label in zip(result.keypoints, result.boxes.cls)
        ]


with make_client(
    host="http://localhost", port=8080, credentials=("user", "password")
) as client:
    cvataa.annotate_task(client, 41617, AnnotShapes)
