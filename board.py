import cv2 as cv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from pathlib import Path

path = Path("resources")

images = ["chess_1.jpg", "chess_2.jpeg"]
images = [cv.imread(str(path / image)) for image in images]


class ImageCanny:
    # low, high = 360, 360
    #low, high = 0, 500
    #low, high = 120, 200
    #low, high = 200, 300
    low, high = 100, 400

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

for index, image in enumerate(images):
    resized = cv.resize(image, (500, 500))
    blur = cv.GaussianBlur(resized, (3, 3), 9)
    cv.imshow('pre_' + str(index), blur)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    imgs.append(
        {
            "original": resized.copy(),
            "resize": resized,
            "gray": gray,
        }
    )

cv.waitKey(0)

def getColor(degree):
    if degree <= 90:
        return (255, 0, 0)

    if degree <= 180:
        return (0, 255, 0)

    return (0, 0, 255)


def getDegree(theta):
    return theta * (180 / np.pi)


def splitLinesGroups(dataframe):
    dif = dataframe.diff()
    idx = dif.theta.idxmax()
    middleTheta = dataframe[dataframe.index == idx].iloc[0].theta
    return (
        dataframe[dataframe.theta < middleTheta],
        dataframe[dataframe.theta >= middleTheta],
    )


hough = []
once = 1

while once == 1:
    # once = 0
    for index, images in enumerate(imgs):
        original = images.get("resize").copy()

        gray_resized = images.get("gray").copy()

        edges = cv.Canny(gray_resized, C.low, C.high)
        lines = cv.HoughLines(edges, H.rho, H.theta, H.th)

        # radon = cv.ximgproc.RadonTransform(
        #     edges, theta=1, start_angle=0, end_angle=360, norm=True
        # )

        dataset = []
        lll = np.empty((0, 2), dtype=np.float32)

        if lines is not None:
            r = []
            t = []
            x = []
            y = []
            for line in lines:
                rho, theta = line[0]
                lll = np.append(lll, [[rho, theta]], axis=0)

                r.append(rho)
                t.append(theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                x.append(x0)
                y.append(y0)

                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                if rho > 0:
                    cv.line(original, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    cv.line(original, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # cv.circle(original, (int(x0), int(y0)), 1, (255, 0, 255), 2)

            # compac, labels, centers = cv.kmeans(lll, 3, criteria=(cv.TERM_CRITERIA_MAX_ITER+cv.TermCriteria_EPS, 20, 1), bestLabels=None, attempts=10, flags=cv.KMEANS_RANDOM_CENTERS)
            # lines1 = lll[labels.ravel() == 0]
            # lines2 = lll[labels.ravel() == 1]
            # lines3 = lll[labels.ravel() == 2]
            lines1, lines2, centers = [], [], []
            # print(lines3)
            hough.append({"rho": r, "theta": t, "x": x, "y": y, "radon": [], 'A': lines1, 'B': lines2, 'center': centers})

            # # # lines1, lines2 = splitLinesGroups(dt)
            # for row in lines1:
            #     a = np.cos(row[1])
            #     b = np.sin(row[1])
            #     x = a * row[0]
            #     y = b * row[0]
            #     x1 = int(x + 1000 * (-b))
            #     y1 = int(y + 1000 * (a))
            #     x2 = int(x - 1000 * (-b))
            #     y2 = int(y - 1000 * (a))

            #     cv.line(original, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # for row in lines2:
            #     a = np.cos(row[1])
            #     b = np.sin(row[1])
            #     x = a * row[0]
            #     y = b * row[0]
            #     x1 = int(x + 1000 * (-b))
            #     y1 = int(y + 1000 * (a))
            #     x2 = int(x - 1000 * (-b))
            #     y2 = int(y - 1000 * (a))

            #     cv.line(original, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv.imshow("chess_" + str(index), edges)
        cv.imshow("chessboard_" + str(index), original)

    if cv.waitKey(10) == ord("q"):
        break


domain = np.arange(-np.pi * 4, np.pi * 4, 0.01)

for index, h in enumerate(hough):
    theta = h.get("theta")
    rho = h.get("rho")

    x = h.get("x")
    y = h.get("y")

    plt.figure()
    plt.axes().set_facecolor("black")
    for z in zip(x, y):
        yyy = [z[0] * np.cos(d) + z[1] * np.sin(d) for d in domain]
        plt.plot(domain, yyy, color=(1, 1, 1, 0.5), linewidth=0.5)

    s = sorted(theta)
    dd = [getDegree(t) for t in theta]
    plt.figure()
    sns.scatterplot(x=theta, y=rho, color=(1, 0, 0, 0.5))

    c = h.get('center')
    sns.scatterplot(x=c[:,1], y=c[:,0])

    # A =h.get('A')
    # plt.figure()
    # sns.scatterplot(x=A[:, 1], y=A[:, 0])
    # B =h.get('B')
    # plt.figure()
    # sns.scatterplot(x=B[:, 1], y=B[:, 0])

    # radon = h.get("radon")
    # cv.imshow("radon_" + str(index), radon)


plt.show(block=False)
cv.waitKey(0)
plt.close('all')
cv.destroyAllWindows()
