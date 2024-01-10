import cv2 as cv
import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/best.pt")


class Board:
    def __init__(self, image, pts=()):
        self.image = image
        self.points = pts

    def _try(self):
        self.outPoints = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

        M = cv.getPerspectiveTransform(self.points, self.outPoints)

        return cv.warpPerspective(self.image, M, (300, 300))


root = Path("datasets/chess/images/train")

src = "datasets/chess/images/train/IMG_2661.jpg"

images = list(root.glob("*.jpg"))

for image in images:
    pred = model.predict(image.resolve(), verbose=False, classes=[0], save=False)

    pts = pred[0].keypoints.xy.tolist()[0]
    pts = np.float32(pts)

    b = Board(cv.imread(str(image.resolve())), pts)

    per = b._try()

    cv.imwrite("try/" + image.name, per)
