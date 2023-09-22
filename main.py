from Board import Board

import cv2 as cv

from pathlib import Path
from time import time

b = Board(True)

p = Path("resources")
images = p.glob("*")

# v = cv.VideoCapture(0)

# while v.isOpened():
#     _, frame = v.read()
#     cv.imshow("me", frame)
#     b.setImage(frame)
#     image = frame
#     try:
#         image = b.process()
#     except:
#         pass
#     cv.imshow("result", image)
#     if cv.waitKey(1) == ord("q"):
#         break

# v.release()

for image in list(images):
    im = cv.imread(str(image))
    b.setImage(im)
    result = b.process()

cv.waitKey(0)
cv.destroyAllWindows()
