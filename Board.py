import numpy as np
import cv2 as cv
from math import sqrt, pow
import random

from sklearn.cluster import AgglomerativeClustering


class Line:
    _SIZE = 3000

    @staticmethod
    def getLineCoords(rho, theta) -> tuple:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + Line._SIZE * (-b))
        y1 = int(y0 + Line._SIZE * (a))
        x2 = int(x0 - Line._SIZE * (-b))
        y2 = int(y0 - Line._SIZE * (a))

        return (x1, y1), (x2, y2)

    @staticmethod
    def printLine(img, line, color=(0, 0, 255)) -> None:
        cv.line(
            img, (int(line[3]), int(line[4])), (int(line[5]), int(line[6])), color, 2
        )

    @staticmethod
    def getBestLines(lines, th=15):
        bestLines = []
        bestLines.append(lines[0])

        lines = lines[1:]

        for l in lines:
            rho, theta, acc = l[0], l[1], l[2]
            it = bestLines.copy()

            add = True

            for index, line in enumerate(it):
                rho1, theta1, acc1 = line[0], line[1], line[2]

                dist = np.sqrt(np.power(rho - rho1, 2) + np.power(theta - theta1, 2))
                if dist < th:
                    if acc > acc1:
                        del bestLines[index]
                    else:
                        add = False
                        break
            if add == True:
                bestLines.append(l)

        return bestLines

    @staticmethod
    def getIntersectionPoint(l1, l2):
        (x11, y11, x12, y12) = l1[3], l1[4], l1[5], l1[6]
        (x21, y21, x22, y22) = l2[3], l2[4], l2[5], l2[6]

        px = (x11 * y12 - y11 * x12) * (x21 - x22) - (x11 - x12) * (
            x21 * y22 - y21 * x22
        )
        py = (x11 * y12 - y11 * x12) * (y21 - y22) - (y11 - y12) * (
            x21 * y22 - y21 * x22
        )

        den = (x11 - x12) * (y21 - y22) - (y11 - y12) * (x21 - x22)
        if den == 0:
            return None

        return (px / den, py / den)


class Canny:
    low = 140
    high = 240

    def setLow(self, value):
        if value <= 0:
            value = 1

        self.low = value

    def setHigh(self, value):
        self.high = value

    def __str__(self) -> str:
        return str(self.low) + ", " + str(self.high)


class Hough:
    rho = 1
    theta = np.pi / 180
    threshold = 100

    def setRho(self, value):
        if value <= 0:
            value = 1
        self.rho = value

    def setTheta(self, value):
        if value <= 0:
            value = 1
        self.theta = value * np.pi / 180

    def setThreshold(self, value):
        if value <= 0:
            value = 1
        self.threshold = value

    def __str__(self) -> str:
        return str(self.rho) + ", " + str(self.theta) + ", " + str(self.threshold)


class Gaussian:
    k_size = 3
    sigma = 3

    def setSigma(self, value):
        if value <= 0:
            value = 1
        self.sigma = value

    def setSize(self, value):
        if value <= 0:
            value = 1
        self.k_size = value

    def __str__(self) -> str:
        return str(self.k_size) + ", " + str(self.sigma)


class Board:
    canny: Canny = None
    hough: Hough = None
    gaussian: Gaussian = None

    image = None
    SIZE = 500

    debug = False

    def __init__(self, debug=False) -> None:
        self.canny = Canny()
        self.hough = Hough()
        self.gaussian = Gaussian()
        self.debug = debug

        if debug:
            cv.namedWindow("Tracks")
            cv.createTrackbar(
                "canny_low", "Tracks", self.canny.low, 1000, self.canny.setLow
            )
            cv.createTrackbar(
                "canny_high", "Tracks", self.canny.high, 1000, self.canny.setHigh
            )

            cv.createTrackbar(
                "hough_rho", "Tracks", self.hough.rho, 100, self.hough.setRho
            )
            cv.createTrackbar(
                "hough_theta", "Tracks", int(self.hough.theta), 360, self.hough.setTheta
            )
            cv.createTrackbar(
                "hough_threshold",
                "Tracks",
                self.hough.threshold,
                500,
                self.hough.setThreshold,
            )

            cv.createTrackbar(
                "gaussian_size",
                "Tracks",
                self.gaussian.k_size,
                1000,
                self.gaussian.setSize,
            )
            cv.createTrackbar(
                "gaussian_sigma",
                "Tracks",
                self.gaussian.sigma,
                1000,
                self.gaussian.setSigma,
            )

    def setImage(self, image) -> None:
        self.image = image

    def process(self):
        if self.image is None:
            raise Exception("Image not set")

        while True:
            resize = cv.resize(self.image, (self.SIZE, self.SIZE))
            gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

            blur = cv.GaussianBlur(
                gray, (self.gaussian.k_size, self.gaussian.k_size), self.gaussian.sigma
            )
            edges = cv.Canny(blur, self.canny.low, self.canny.high)

            lines = cv.HoughLinesWithAccumulator(
                edges,
                self.hough.rho,
                self.hough.theta,
                self.hough.threshold,
                min_theta=-np.pi / 2,
                max_theta=np.pi / 2,
            )

            if lines is None or len(lines) < 2:
                return resize

            l = np.empty((0, 7))
            for ll in lines:
                rho, theta, _ = ll[0]
                coords = Line.getLineCoords(rho, theta)

                l = np.append(
                    l,
                    [
                        [
                            *ll[0],
                            coords[0][0],
                            coords[0][1],
                            coords[1][0],
                            coords[1][1],
                        ]
                    ],
                    axis=0,
                )

            lines = l
            imgLines = np.zeros((500, 500, 3))
            for ll in l:
                Line.printLine(imgLines, ll)

            length = len(lines)
            diffMatrix = np.zeros((length, length))
            for i, a in enumerate(lines):
                for j, b in enumerate(lines):
                    theta1, theta2 = a[1], b[1]

                    if theta1 > theta2:
                        diff = theta1 - theta2
                    else:
                        diff = theta2 - theta1

                    diff = abs(diff)
                    ninety = np.pi / 2

                    if ninety > diff:
                        diff = ninety - diff
                    else:
                        diff = diff - ninety
                    diff = ninety - diff

                    diffMatrix[i, j] = abs(diff)

            cluster = AgglomerativeClustering(
                metric="precomputed", linkage="single"
            ).fit(diffMatrix)

            labels = cluster.labels_

            h = lines[labels == 0]
            v = lines[labels == 1]

            coloredLines = np.zeros((500, 500, 3))
            for line in h:
                Line.printLine(coloredLines, line)

            for line in v:
                Line.printLine(coloredLines, line, (255, 0, 0))

            bestLines = np.zeros((500, 500, 3))
            hs = Line.getBestLines(h)
            vs = Line.getBestLines(v)

            verticals, horizontals = np.zeros((500, 500, 3)), np.zeros((500, 500, 3))

            middleHorizontal = (0, 0, 0, 0, 250, 500, 250)
            middleVertical = (0, 0, 0, 250, 0, 250, 500)

            Line.printLine(horizontals, middleVertical, (255, 0, 0))
            Line.printLine(verticals, middleHorizontal, (255, 0, 0))

            horizontalPivotPoint = (0, 250)
            verticalPivotPoint = (250, 0)

            hs_np = np.array(hs)
            vs_np = np.array(vs)

            hs_np = np.hstack((hs_np, np.zeros((hs_np.shape[0], 3), dtype=hs_np.dtype)))
            vs_np = np.hstack((vs_np, np.zeros((vs_np.shape[0], 3), dtype=vs_np.dtype)))

            for line in hs_np:
                Line.printLine(bestLines, line)
                Line.printLine(resize, line)
                center = Line.getIntersectionPoint(middleVertical, line)
                line[7] = int(center[0])
                line[8] = int(center[1])
                line[9] = int(
                    sqrt(
                        pow(line[7] - verticalPivotPoint[0], 2)
                        + pow(line[8] - verticalPivotPoint[1], 2)
                    )
                )

            for line in vs_np:
                Line.printLine(bestLines, line, (255, 0, 0))
                Line.printLine(resize, line, (255, 0, 0))
                center = Line.getIntersectionPoint(middleHorizontal, line)
                line[7] = int(center[0])
                line[8] = int(center[1])
                line[9] = int(
                    sqrt(
                        pow(line[7] - horizontalPivotPoint[0], 2)
                        + pow(line[8] - horizontalPivotPoint[1], 2)
                    )
                )

            hs_np = hs_np[hs_np[:, 9].argsort()]
            vs_np = vs_np[vs_np[:, 9].argsort()]

            for line in hs_np:
                Line.printLine(horizontals, line, (255, 255, 255))

            for line in vs_np:
                Line.printLine(verticals, line, (255, 255, 255))

            inter = np.zeros((500, 500, 3))
            centers = []

            inter_np = np.zeros((hs_np.shape[0], vs_np.shape[0], 2))

            for h_idx, h in enumerate(hs_np):
                for v_idx, v in enumerate(vs_np):
                    center = Line.getIntersectionPoint(h, v)
                    centers.append(center)

                    inter_np[h_idx, v_idx] = [center[0], center[1]]

                    if center is not None:
                        cv.drawMarker(
                            inter,
                            (int(center[0]), int(center[1])),
                            (255, 0, 255),
                            markerType=cv.MARKER_CROSS,
                            markerSize=5,
                            thickness=3,
                        )
                        cv.drawMarker(
                            resize,
                            (int(center[0]), int(center[1])),
                            (255, 0, 255),
                            markerType=cv.MARKER_CROSS,
                            markerSize=5,
                            thickness=3,
                        )

            s = inter_np.shape
            x0, y0 = s[0], s[1]

            squares = []

            squares_overlay = resize.copy()

            for i in range(0, x0 - 1):
                for j in range(0, y0 - 1):

                    sq = (
                        inter_np[i][j],
                        inter_np[i + 1][j],
                        inter_np[i + 1][j + 1],
                        inter_np[i][j + 1],
                    )

                    ran = random.Random()

                    r, g, b = (
                        ran.randint(0, 255),
                        ran.randint(0, 255),
                        ran.randint(0, 255),
                    )

                    cv.fillConvexPoly(
                        squares_overlay,
                        np.array(
                            [
                                [
                                    [sq[0][0], sq[0][1]],
                                    [sq[1][0], sq[1][1]],
                                    [sq[2][0], sq[2][1]],
                                    [sq[3][0], sq[3][1]],
                                ]
                            ],
                            dtype=np.int32,
                        ),
                        (b, g, r),
                    )
                    squares.append(sq)

            final_square = cv.addWeighted(squares_overlay, 0.4, resize, 1 - 0.4, 0)

            angles = l[:, 1]
            angles = np.sort(angles)
            grad = np.gradient(angles)

            if self.debug:
                cv.imshow("Canny", edges)
                cv.imshow("Lines", imgLines)
                cv.imshow("Vertical-Horizontal", coloredLines)
                cv.imshow("Best Vertical-Horizontal", bestLines)
                cv.imshow("Intersections", inter)
                cv.imshow("Vertical Lines", verticals)
                cv.imshow("Horizontal Lines", horizontals)
                cv.imshow("Squares", final_square)
                cv.imshow("Result", resize)

            if cv.waitKey(100) == ord("q") or not self.debug:
                break

        return resize, centers, (hs, vs)
