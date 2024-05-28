import cv2 as cv
import numpy as np
from Board import Board
from pathlib import Path
from ultralytics import YOLO
from Perspective import Perspective
from ChessPiece import ChessPiece
from utils import getBoardKeypoints, convertToPx

model = YOLO("models/best.pt")

image = Path("datasets/chess/images/train/test.JPG")

prediction = model.predict(image.resolve(), verbose=False, save=False)[0]

classes, keypoints = prediction.boxes.cls, prediction.keypoints.xy

boardKeypoints = None

for c, k in zip(classes, keypoints):
    if c.item() == 0:
        boardKeypoints = np.float32(k.tolist())
        break

# -> DEBUG <-

kpts = getBoardKeypoints(image)
im = cv.imread(str(image.resolve()))
kkk = convertToPx(im, kpts)

kkk = list(map(float, kkk))
kkk = np.array(kkk)
kkk = kkk.reshape((4, 2))
kkk = np.float32(kkk.tolist())
boardKeypoints = kkk

# -> DEBUG <-

if boardKeypoints is None:
    print("No chessboard detected")
    exit(0)

im = cv.imread(str(image.resolve()))
squares_overlay = im.copy()

perspective = Perspective(im, boardKeypoints)
boardPerspective = perspective.apply()

board = Board(debug=False)
board.setImage(boardPerspective)

try:
    boardResized, centers, (horizontal, vertical), squares = board.process()
except:
    print("It was not possible to process the board")
    exit(0)

boardSquares = perspective.undoPerspective(np.array(squares))

centroids, chessboard = [], []

pieces = []

for c, k in zip(classes, keypoints):

    if c.item() == 0:
        continue

    piece = ChessPiece()
    piece.setClass(int(c.item()))

    for idx, i in enumerate(k):
        piece.addPoint(i)

    centroid = piece.getCentroid()
    centroids.append(centroid)

for sq in boardSquares:
    item = []
    item.append(sq)
    for c in centroids:
        p = np.array(sq, dtype=np.float32)
        if len(item) == 1 and cv.pointPolygonTest(p, c, False) > 0:
            item.append(c)
    if len(item) == 1:
        item.append(0)
    chessboard.append(item)

for sq, c in chessboard:
    if c == 0:
        cv.fillConvexPoly(
            squares_overlay,
            np.array(
                [
                    [
                        [sq[0][0], sq[0][1]],
                        [sq[1][0], sq[1][1]],
                        [sq[2][0], sq[2][1]],
                        [sq[3][0], sq[3][1]],
                    ]
                ],
                dtype=np.int32,
            ),
            (0, 0, 255),
        )
        continue

    cv.drawMarker(im, (int(c[0]), int(c[1])), (255, 0, 255), cv.MARKER_DIAMOND, 15, 15)

    cv.fillConvexPoly(
        squares_overlay,
        np.array(
            [
                [
                    [sq[0][0], sq[0][1]],
                    [sq[1][0], sq[1][1]],
                    [sq[2][0], sq[2][1]],
                    [sq[3][0], sq[3][1]],
                ]
            ],
            dtype=np.int32,
        ),
        (255, 255, 0),
    )

for idx, p in enumerate(boardKeypoints):
    cv.drawMarker(im, (int(p[0]), int(p[1])), (0, 0, 255), cv.MARKER_CROSS, 10, 5)

im = cv.addWeighted(squares_overlay, 0.4, im, 1 - 0.4, 0)

cv.imwrite("board/" + image.name, im)

cv.namedWindow("markers", cv.WINDOW_GUI_EXPANDED)
while True:
    cv.imshow("markers", im)
    if cv.waitKey(100) == ord("q"):
        break

print("Press any key to exit...")
cv.waitKey(0)
