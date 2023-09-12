import cv2 as cv

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from math import sqrt

from pathlib import Path

path = Path("resources")

images = ["chess_1.jpg", 'chess_2.jpeg']
images = [cv.imread(str(path / image)) for image in images]


class Tracks:

    gaussian = 3
    sigma = 3

    rho, theta, th = 1, 1, 100

    c_low, c_high = 100, 200

    def setGaussian(self, v):
        if v % 2 == 0:
            v = v +1
        self.gaussian = v

    def setSigma(self, v):
        self.sigma = v

    def setRho(self, v):
        if v <= 0:
            v = 1
        self.rho = v
    def setTheta(self, v):
        if v <= 0:
            v = 1
        self.theta = v*np.pi/180
    def setTh(self, v):
        if v <= 0:
            v= 1
        self.th = v

    def setCLow(self, v):
        self.c_low = v
    def setCHigh(self, v):
        self.c_high = v



cv.namedWindow('tracks')

t = Tracks()

cv.createTrackbar('gaussian', 'tracks', t.gaussian, 50, t.setGaussian)
cv.createTrackbar('sigma', 'tracks', t.sigma, 50, t.setSigma)

cv.createTrackbar('c_low', 'tracks', t.c_low, 1000, t.setCLow)
cv.createTrackbar('c_high', 'tracks', t.c_high, 1000, t.setCHigh)

cv.createTrackbar('rho', 'tracks', t.rho, 10, t.setRho)
cv.createTrackbar('theta', 'tracks', t.theta, 360, t.setTheta)
cv.createTrackbar('th', 'tracks', t.th, 500, t.setTh)

hs = []
vs = []

once = 1
while once == 1:
    once = 1
    for index, image in enumerate(images):

        h= []
        v=[]

        i = image.copy()
        i = cv.resize(i, (500, 500))
        g = cv.cvtColor(i, cv.COLOR_BGR2GRAY)

        black = np.zeros((500, 500, 3))

        blur = cv.GaussianBlur(g, (t.gaussian, t.gaussian), t.sigma)

        c = cv.Canny(blur, t.c_low, t.c_high)
        cv.imshow('canny_' + str(index), c)

        lines = cv.HoughLines(c, t.rho, t.theta, t.th)

        if lines is not None:

            for l in lines:
                rho, theta = l[0]

                

                distToOrigin = theta
                distToEnd = sqrt(pow(np.pi-theta,2))
                distToMiddle = sqrt(pow(np.pi/2 - theta, 2))

                #cv.line(i, (x1, y1), (x2, y2), (255, 0, 0), 1)
                add = True
                if distToMiddle < distToOrigin and distToMiddle < distToEnd:

                    if len(h) == 0:
                        h.append([rho, theta])

                    for l in h:
                        dist = sqrt(pow((l[0]-rho) + (l[1]-theta), 2))

                        if dist <= 16:
                            add = False
                            break
                    
                    if add == True:
                        h.append([rho, theta])
                else:

                    if len(v) == 0:
                        v.append([rho, theta])

                    for l in v:
                        dist = sqrt(pow((l[0]-rho) + (l[1]-theta), 2))

                        if dist <= 16:
                            add = False
                            break
                    if add == True:
                        v.append([rho, theta])

                for l in h:
                    a = np.cos(l[1])
                    b = np.sin(l[1])
                    x0 = a * l[0]
                    y0 = b * l[0]

                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv.line(black, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.line(i, (x1, y1), (x2, y2), (0, 0, 255), 1)

                for l in v:
                    a = np.cos(l[1])
                    b = np.sin(l[1])
                    x0 = a * l[0]
                    y0 = b * l[0]

                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv.line(black, (x1, y1), (x2, y2), (255, 0,0), 2)
                    cv.line(i, (x1, y1), (x2, y2), (255, 0,0), 1)
                
        cv.imshow('original_' + str(index), i)
        cv.imshow('lines_' + str(index), black)

        result = blur
        cv.imshow('chess_' + str(index), result)


        hs.append(h)
        vs.append(v)

    if cv.waitKey(10) == ord('q'):
        break


cv.destroyAllWindows()