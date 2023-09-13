from pathlib import Path

import cv2 as cv
import numpy as np

path = Path("resources")

image = "chess_2.jpeg"

image = cv.imread(str(path / image), cv.IMREAD_COLOR)

resize = cv.resize(image, (500, 500))
gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3, 3), 3)
edges = cv.Canny(blur, 100, 200)
lines = cv.HoughLinesWithAccumulator(edges, 1, np.pi/180, 80)

h, v = [],[]

def poi(x1, y1, x2, y2, x3,y3,x4,y4):
    px = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
    py = (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)

    den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if den == 0:
        return None
    
    return (px/den, py/den)

def getLineCoords(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 3000 * (-b))
    y1 = int(y0 + 3000 * (a))
    x2 = int(x0 - 3000 * (-b))
    y2 = int(y0 - 3000 * (a))

    return (x1,y1), (x2,y2)


def getBestLines(lines, th = 15):

    bestLines = []
    bestLines.append(lines[0])
    del lines[0]

    for l in lines:

        rho, theta, acc = l
        it = bestLines.copy()

        add = True

        for index, line in enumerate(it):
            rho1, theta1, acc1 = line

            dist = np.sqrt(np.power((rho-rho1)+(theta-theta1),2))

            if dist < th:

                if acc > acc1:
                    del bestLines[index]
                else:
                    add = False
                    break
        if add == True:
            bestLines.append(l)
    return bestLines

def printLines(lines, image, c):
    for l in lines:
        points = getLineCoords(l[0], l[1])
        cv.line(image, points[0], points[1], c, 1)


for l in lines:
    theta = l[0][1]

    distToOrigin = theta
    distToEnd = np.sqrt(np.power(np.pi-theta,2))
    distToMiddle = np.sqrt(np.power(np.pi/2 - theta, 2))

    if distToMiddle < distToOrigin and distToMiddle < distToEnd:
        h.append(*l)
    else:
        v.append(*l)

black = np.zeros((500, 500, 3))
hs = getBestLines(h)
vs = getBestLines(v)

for v in vs:
    for h in hs:

        hline, vline = getLineCoords(h[0], h[1]), getLineCoords(v[0], v[1])
        point = poi(*hline[0], *hline[1], *vline[0], *vline[1])

        if point is not None:
            cv.drawMarker(resize, (int(point[0]), int(point[1])), (0,0,255), markerType=cv.MARKER_DIAMOND, markerSize=8, thickness=2)

c = (0, 0, 0)
printLines(hs, resize, (0, 0, 255))
printLines(vs, resize, (255, 0, 0))
cv.imshow('Result', resize)

cv.waitKey(0)
cv.destroyAllWindows()
