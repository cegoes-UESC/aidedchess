from Board import Board

import cv2 as cv

from pathlib import Path

b = Board(False)

p = Path("resources")

images = p.glob("*")

images = [cv.imread(str(i)) for i in list(images)]

for image in images:
    b.setImage(image)
    b.process()
