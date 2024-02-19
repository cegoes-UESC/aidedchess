import cv2 as cv
import numpy as np
from Board import Board
from pathlib import Path
from ultralytics import YOLO
from Perspective import Perspective

model = YOLO("models/best.pt")


image = Path("datasets/chess/images/train/IMG_2610.jpg")

prediction = model.predict(image.resolve(), verbose=False, classes=[0], save=False)

boardKeypoints = np.float32(prediction[0].keypoints.xy.tolist()[0])

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
