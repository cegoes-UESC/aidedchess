import cv2 as cv
import numpy as np
from Board import Board
from pathlib import Path
from ultralytics import YOLO
from Perspective import Perspective
from ChessPiece import ChessPiece
from utils import getBoardKeypoints, convertToPx
from ChessBoard import ChessBoard, ChessBoardCell, ChessBoardData, CellState

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

pieces, chessboard = [], []

for c, k in zip(classes, keypoints):

    if c.item() == 0:
        continue

    piece = ChessPiece()
    piece.setClass(int(c.item()))

    for idx, i in enumerate(k):
        piece.addPoint(i.tolist())

    pieces.append(piece)

bs = np.array(boardSquares)
bs = bs.reshape((8, 8, 4, 2))

chessboard = ChessBoard()

for i, l in enumerate(bs):
    for j, r in enumerate(l):

        cell = ChessBoardCell(r)

        cellData = ChessBoardData(cell)

        chessboard.addData(i, j, cellData)

        for piece in pieces:
            centroid = piece.getCentroid()
            cell_f32 = np.array(r, dtype=np.float32)
            if cv.pointPolygonTest(cell_f32, centroid, False) > 0:
                cellData = chessboard.getBoardData(i, j)
                if cellData.piece is not None:
                    print("=> Celldata already has a Piece, ignoring")
                    continue
                else:
                    cell.state = CellState.OCCUPIED
                    cellData.piece = piece


for idx, p in enumerate(boardKeypoints):
    cv.drawMarker(im, (int(p[0]), int(p[1])), (0, 0, 255), cv.MARKER_CROSS, 10, 5)

chessboard.draw(squares_overlay)
im = cv.addWeighted(squares_overlay, 0.4, im, 1 - 0.4, 0)

cv.imwrite("board/" + image.name, im)

cv.namedWindow("markers", cv.WINDOW_GUI_EXPANDED)
while True:
    cv.imshow("markers", im)
    if cv.waitKey(100) == ord("q"):
        break

print("Press any key to exit...")
cv.waitKey(0)
