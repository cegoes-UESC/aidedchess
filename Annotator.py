import PIL
from typing import List
from cvat_sdk import models
from ultralytics import YOLO
from cvat_sdk import make_client
import cvat_sdk.auto_annotation as cvataa


class AnnotShapes:
    def __init__(self) -> None:
        self.model = YOLO("models/pose.pt")

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.skeleton_label_spec(
                    name,
                    id,
                    [
                        models.SublabelRequest(str(x), id=x, type="points")
                        for x in range(1, 5)
                    ],
                )
                for id, name in self.model.names.items()
            ],
        )

    def detect(
        self, context, image: PIL.Image.Image
    ) -> List[models.LabeledShapeRequest]:
        print(f"=== Annotating image {context.frame_name} ===")
        results = self.model(image, verbose=False)

        return [
            cvataa.skeleton(
                int(label.item()),
                elements=[
                    {
                        "frame": 0,
                        "type": "points",
                        "label_id": idx + 1,
                        "points": sk.tolist(),
                    }
                    for idx, sk in enumerate(skeleton.xy[0])
                ],
            )
            for result in results
            for skeleton, label in zip(result.keypoints, result.boxes.cls)
        ]


import sys

try:
    taskId = sys.argv[1]
except:
    print("Unable to read task ID")
    exit()


with make_client(
    host="http://localhost", port=8080, credentials=("almir", "1234")
) as client:
    print(f"Running AUTO ANNOTATION on TASK {taskId}")
    cvataa.annotate_task(client, int(taskId), AnnotShapes(), clear_existing=True)
