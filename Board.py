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
        self.outPoints = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])

        M = cv.getPerspectiveTransform(self.points, self.outPoints)
        self.trans = M

        return cv.warpPerspective(self.image, M, (500, 500))

    def un(self, inn):
        w, h = self.image.shape[1], self.image.shape[0]
        return cv.warpPerspective(inn, self.trans, (w, h), flags=cv.WARP_INVERSE_MAP)


root = Path("datasets/chess/images/train")

image = Path("datasets/chess/images/train/IMG_2651.jpg")

images = list(root.glob("*.jpg"))

# for image in images:
pred = model.predict(image.resolve(), verbose=False, classes=[0], save=False)

pts = pred[0].keypoints.xy.tolist()[0]
pts = np.float32(pts)

b = Board(cv.imread(str(image.resolve())), pts)

per = b._try()

from _Board import Board as B

bbb = B(debug=True)

bbb.setImage(per)

res = bbb.process()

cv.namedWindow("resized", cv.WINDOW_GUI_EXPANDED)
cv.imshow("resized", res)

cv.imshow("try", per)

oooooo = b.un(res)

cv.namedWindow("ooo", cv.WINDOW_GUI_EXPANDED)
cv.imshow("ooo", oooooo)


im = cv.imread(str(image.resolve()))

cv.namedWindow("r", cv.WINDOW_GUI_EXPANDED)
cv.imshow("r", im)

cv.waitKey(0)
