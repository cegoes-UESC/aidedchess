import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("pose-14-12_13.pt")


class Board:
    def __init__(self, image, pts=()):
        self.image = image
        self.points = pts

    def _try(self):
        self.outPoints = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

        M = cv.getPerspectiveTransform(self.points, self.outPoints)

        return cv.warpPerspective(self.image, M, (300, 300))


src = "IMG_2481.jpg"

cv.namedWindow("SRC", cv.WINDOW_GUI_EXPANDED)
image = cv.imread(src)

cv.imshow("SRC", image)

pred = model.predict(src, verbose=False, classes=[0], save=False)

pts = pred[0].keypoints.xy.tolist()[0]
pts = np.float32(pts)

b = Board(image, pts)

per = b._try()

cv.imshow("PER", per)
cv.waitKey(0)
