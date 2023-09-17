from Board import Board

import cv2 as cv

from pathlib import Path
from time import time

b = Board(False)

p = Path("resources")

images = p.glob("*")

images = [cv.imread(str(i)) for i in list(images)]

v = cv.VideoCapture(0)

while v.isOpened():
    _, frame = v.read()
    cv.imshow("me", frame)
    b.setImage(frame)
    image = frame
    try:
        image = b.process()
    except:
        pass
    cv.imshow("result", image)
    if cv.waitKey(1) == ord("q"):
        break


v.release()
cv.waitKey(0)
cv.destroyAllWindows()

# for image in images:
#     b.setImage(image)
#     b.process()
