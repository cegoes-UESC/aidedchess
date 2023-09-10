import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

from pathlib import Path

path = Path("resources")

images = ["chess_1.jpg", "chess_2.jpeg"]
images = [cv.imread(str(path / image)) for image in images]


class ImageCanny:
    #low, high = 360, 360
    low, high = 0, 600

    def setLow(self, l):
        self.low = l

    def setHigh(self, h):
        self.high = h


class ImageHough:
    # rho, theta, th = 1, 1, 85
    rho, theta, th = 1, 1, 100

    def setRho(self, r):
        if r <= 0:
            r = 1
        self.rho = r

    def setTheta(self, theta):
        self.theta = theta * (np.pi / 180)

    def setTh(self, th):
        self.th = th


cv.namedWindow("chessboard_0")
cv.namedWindow("chessboard_1")
cv.namedWindow("tracks")

C = ImageCanny()
H = ImageHough()

cv.createTrackbar("low_canny", "tracks", C.low, 1000, C.setLow)
cv.createTrackbar("high_canny", "tracks", C.high, 1000, C.setHigh)

cv.createTrackbar("rho", "tracks", H.rho, 50, H.setRho)
cv.createTrackbar("theta", "tracks", H.theta, 360, H.setTheta)
cv.createTrackbar("th", "tracks", H.th, 500, H.setTh)

imgs = []

for image in images:
    i = cv.GaussianBlur(image, (3, 3), 0)
    resized = cv.resize(i, (500, 500))
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    imgs.append(
        {
            "original": resized.copy(),
            "resize": resized,
            "gray": gray,
        }
    )


def getColor(degree):
    if degree <= 45:
        return (255, 0, 0)

    if degree <= 90:
        return (0, 255, 0)

    return (0, 0, 255)


def getDegree(theta):
    return theta * (180 / np.pi)


hough = []
once = 1

while once == 1:
    # once = 0
    for index, images in enumerate(imgs):
        original = images.get("resize").copy()

        gray_resized = images.get("gray").copy()

        edges = cv.Canny(gray_resized, C.low, C.high)
        lines = cv.HoughLines(edges, H.rho, H.theta, H.th)

        if lines is not None:
            r = []
            t = []
            xx = []
            yy = []
            for line in lines:
                rho, theta = line[0]
                r.append(rho)
                t.append(theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                xx.append(x0)
                yy.append(y0)

                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv.circle(original, (int(x0), int(y0)), 1, (255, 0, 255), 2)
                cv.line(original, (x1, y1), (x2, y2), getColor(getDegree(theta)), 1)

            hough.append({"rho": r, "theta": t, "x": xx, "y": yy})

        cv.imshow("chess_" + str(index), edges)
        cv.imshow("chessboard_" + str(index), original)

    if cv.waitKey(50) == ord("q"):
        break


domain = np.arange(-np.pi, np.pi, 0.05)

for h in hough:
    theta = h.get("theta")
    rho = h.get("rho")

    plt.figure()
    x = h.get("x")
    y = h.get("y")
    plt.scatter(x, y, s=rho)

    plt.figure()
    plt.axes().set_facecolor("black")
    for z in zip(x, y):
        yyy = [z[0] * np.cos(d) + z[1] * np.sin(d) for d in domain]
        plt.plot(domain, yyy, color=(1, 1, 1, 0.5), linewidth=0.5)

    s = sorted(theta)
    dd = [getDegree(t) for t in theta]
    plt.scatter(theta, rho, color=(1, 0, 0, 0.5))


plt.show()

cv.waitKey(0)
