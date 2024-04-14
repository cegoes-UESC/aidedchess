import cv2 as cv
import numpy as np
from Board import Board
from pathlib import Path
from ultralytics import YOLO
from Perspective import Perspective

model = YOLO("models/pose.pt")


image = Path("images/IMG_2658.jpg")

prediction = model.predict(image.resolve(), verbose=False, save=False)[0]


classes, keypoints = prediction.boxes.cls, prediction.keypoints.xy

for c, k in zip(classes, keypoints):
    if c.item() == 0:
        boardKeypoints = np.float32(k.tolist())
        continue
    print(c.item(), k)

im = cv.imread(str(image.resolve()))

perspective = Perspective(im, boardKeypoints)
boardPerspective = perspective.apply()

board = Board(debug=True)
board.setImage(boardPerspective)

boardResized, centers, (horizontal, vertical) = board.process()
boardIntersections = perspective.undoPerspective(np.array([centers]))

intersections = boardIntersections[0]

for point in intersections:
    cv.drawMarker(
        im,
        (int(point[0]), int(point[1])),
        (255, 0, 255),
        cv.MARKER_CROSS,
        50,
        3,
    )


cv.namedWindow("markers", cv.WINDOW_GUI_EXPANDED)
while True:
    cv.imshow("markers", im)
    if cv.waitKey(100) == ord("q"):
        break

print("Press any key to exit...")
cv.waitKey(0)
