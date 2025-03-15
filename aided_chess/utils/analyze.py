from pathlib import Path

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering

path = Path("resources")

image = "chess_5.jpg"

image = cv.imread(str(path / image), cv.IMREAD_COLOR)

resize = cv.resize(image, (500, 500))
gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3, 3), 3)
edges = cv.Canny(blur, 100, 200)
lines = cv.HoughLinesWithAccumulator(edges, 1, np.pi / 180, 80)

h, v = [], []

cv.imshow("original", resize)


def poi(x1, y1, x2, y2, x3, y3, x4, y4):
    px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None

    return (px / den, py / den)


def getLineCoords(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 3000 * (-b))
    y1 = int(y0 + 3000 * (a))
    x2 = int(x0 - 3000 * (-b))
    y2 = int(y0 - 3000 * (a))

    return (x1, y1), (x2, y2)


def getBestLines(lines, th=15):
    bestLines = []
    bestLines.append(lines[0])
    del lines[0]

    for l in lines:
        rho, theta, acc = l
        it = bestLines.copy()

        add = True

        for index, line in enumerate(it):
            rho1, theta1, acc1 = line

            dist = np.sqrt(np.power((rho - rho1) + (theta - theta1), 2))
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


size = len(lines)

arr = np.empty((size, size))


for i in range(size):
    for j in range(size):
        iii = np.zeros((500, 500, 3))
        l1 = lines[i][0]
        l2 = lines[j][0]

        rho, theta, _ = l1
        rho2, theta2, _ = l2

        print("theta:", theta)
        print("theta2:", theta2)

        # line1 = getLineCoords(rho, theta)
        # line2 = getLineCoords(rho2, theta2)

        # slope1 = getSlope(line1[0], line1[1])
        # slope2 = getSlope(line2[0], line2[1])

        cv.putText(
            iii,
            "Theta 1" + str(theta),
            (10, 30),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
        )

        cv.putText(
            iii,
            "Theta 2" + str(theta2),
            (10, 60),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
        )

        printLines([l1], iii, (255, 0, 0))
        printLines([l2], iii, (0, 255, 0))

        t, t2 = abs(theta), abs(theta2)
        if t > t2:
            diff = t - t2
        else:
            diff = t2 - t

        # diff1 = theta - theta2
        # diff2 = theta2 - theta

        # if diff1 < diff2:
        #     diff = diff1
        # else:
        #     diff = diff2

        # diff = abs(theta - theta2)
        print("diff:", diff)

        cv.putText(
            iii,
            "Diff " + str(diff),
            (10, 90),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
        )

        cv.imshow("Line", iii)
        # cv.waitKey(0)

        arr[i, j] = diff

print(arr)

cv.imshow("asdwqdq", arr)

clustering = AgglomerativeClustering(metric="precomputed", linkage="single").fit(arr)

# print(clustering)

print(clustering.n_clusters_)

l = clustering.labels_

lul = np.array(lines)

v = lul[l == 0]
h = lul[l == 1]

zzz = np.zeros((500, 500, 3))

print(h[0])

print(len(v))
print(len(h))

vv = []
hh = []

for i in v:
    vv.append(*i)

for i in h:
    hh.append(*i)

v = vv
h = hh

printLines(v, zzz, (0, 0, 255))
printLines(h, zzz, (255, 0, 0))

cv.imshow("vai", zzz)
cv.waitKey(0)

lll = []
for l in lines:
    lll.append(*l)

lll = np.array(lll)

print(lll[abs(lll[:, 1]) < np.pi / 4])

# middle = lll[np.abs(lll[:,1]) < np.pi/4]
# print(middle)


# for l in lines:
#     theta = l[0][1]

#     distToOrigin = np.abs(theta)
#     distToStart = np.sqrt(np.power(-np.pi / 2 - theta, 2))
#     distToEnd = np.sqrt(np.power(np.pi / 2 - theta, 2))

#     if distToStart < distToOrigin or distToEnd < distToOrigin:
#         h.append(*l)
#     else:
#         v.append(*l)

black = np.zeros((500, 500, 3))
hs = getBestLines(h)
vs = getBestLines(v)

for v in vs:
    for h in hs:
        hline, vline = getLineCoords(h[0], h[1]), getLineCoords(v[0], v[1])
        point = poi(*hline[0], *hline[1], *vline[0], *vline[1])

        if point is not None:
            cv.drawMarker(
                resize,
                (int(point[0]), int(point[1])),
                (0, 0, 255),
                markerType=cv.MARKER_DIAMOND,
                markerSize=8,
                thickness=2,
            )

c = (0, 0, 0)
printLines(hs, resize, (0, 0, 255))
printLines(vs, resize, (255, 0, 0))
cv.imshow("Result", resize)

domain = np.arange(-np.pi / 2, np.pi / 2, 0.01)

v = np.array(vs)
h = np.array(hs)

res = np.vstack((v, h))

plt.figure()
plt.axes().set_facecolor("black")
for l in res:
    a = np.cos(l[1])
    b = np.sin(l[1])

    x0 = a * l[0]
    y0 = b * l[0]
    yyy = [x0 * np.cos(d) + y0 * np.sin(d) for d in domain]
    sns.lineplot(x=domain, y=yyy, c=(1, 1, 1, 0.5))

sns.scatterplot(x=res[:, 1], y=res[:, 0])
plt.figure()

sns.scatterplot(x=v[:, 1], y=v[:, 0], color="red")
sns.scatterplot(x=h[:, 1], y=h[:, 0], color="blue")
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
