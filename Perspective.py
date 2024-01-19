import cv2 as cv
import numpy as np


class Perspective:
    def __init__(self, image, pts=(), size=500):
        self.image = image
        self.points = pts
        self.matrix = []
        self.size = size

    def apply(self):
        self.outPoints = np.float32(
            [[0, 0], [self.size, 0], [self.size, self.size], [0, self.size]]
        )

        self.matrix = cv.getPerspectiveTransform(self.points, self.outPoints)

        return cv.warpPerspective(self.image, self.matrix, (self.size, self.size))

    def ll(self, points):
        # w, h = self.image.shape[1], self.image.shape[0]
        inverseMatrix = np.linalg.inv(self.matrix)
        return cv.perspectiveTransform(points, inverseMatrix)
        # return cv.warpPerspective(inn, self.trans, (w, h), flags=cv.WARP_INVERSE_MAP)
