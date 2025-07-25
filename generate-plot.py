from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO


from aided_chess.models import (
    Perspective,
    ChessPiece,
    ChessBoard,
    ChessBoardCell,
    ChessBoardData,
    CellState,
    Board,
)
from aided_chess.models.board import Line
from aided_chess.utils.utils import getBoardKeypoints, convertToPx


def scale(boardKeypoints, scale=1.01):
    p1, p3 = boardKeypoints[0], boardKeypoints[2]
    p2, p4 = boardKeypoints[1], boardKeypoints[3]

    l1, l2 = (0, 0, 0, p1[0], p1[1], p3[0], p3[1]), (
        0,
        0,
        0,
        p2[0],
        p2[1],
        p4[0],
        p4[1],
    )
    center = Line.getIntersectionPoint(l1, l2)

    def scale_vector(
        point: tuple[int, int], reference: tuple[int, int]
    ) -> tuple[int, int]:

        vector = point[0] - reference[0], point[1] - reference[1]
        vector = vector[0] * scale, vector[1] * scale
        return vector[0] + reference[0], vector[1] + reference[1]

    v1, v2, v3, v4 = (
        scale_vector(p1, center),
        scale_vector(p2, center),
        scale_vector(p3, center),
        scale_vector(p4, center),
    )

    boardKeypoints[0] = v1
    boardKeypoints[1] = v2
    boardKeypoints[2] = v3
    boardKeypoints[3] = v4

    return boardKeypoints


model = YOLO("pose_models/final.pt")

image = Path("datasets/chess/images/train/IMG_2831.JPG")

prediction = model.predict(image.resolve(), verbose=False, save=False)[0]

classes, keypoints = prediction.boxes.cls, prediction.keypoints.xy


boardKeypoints = None

kpts = getBoardKeypoints(image)
im = cv.imread(str(image.resolve()))
perfect_keypoints = convertToPx(im, kpts)

perfect_keypoints = list(map(float, perfect_keypoints))
perfect_keypoints = np.array(perfect_keypoints)
perfect_keypoints = perfect_keypoints.reshape((4, 2))
perfect_keypoints = np.float32(perfect_keypoints.tolist())
boardKeypoints = perfect_keypoints


im = cv.imread(str(image.resolve()))
orig = im.copy()
squares_overlay = im.copy()

while True:

    perspective = Perspective(im, boardKeypoints)
    boardPerspective = perspective.apply()

    board = Board(debug=False)
    board.setImage(boardPerspective)
    board.setImageName(image.name)

    try:
        boardResized, centers, (horizontal, vertical), squares = board.process()

        print(len(horizontal), len(vertical))

        if len(horizontal) != 9 or len(vertical) != 9:
            boardKeypoints = scale(boardKeypoints)
            continue
        else:
            break
    except Exception as e:
        raise e


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

boardSquares = np.array(boardSquares)
boardSquares = boardSquares.reshape((8, 8, 4, 2))

to_draw = boardSquares.reshape((64, 4, 2))

points = orig.copy()

for d in to_draw:
    for s in d:
        cv.drawMarker(
            points, (int(s[0]), int(s[1])), (0, 0, 255), cv.MARKER_DIAMOND, 20, 15
        )

chessboard = ChessBoard(boardKeypoints)

for i, _ in enumerate(boardSquares):
    for j, square in enumerate(_):

        cell = ChessBoardCell(square)

        cellData = ChessBoardData(cell)

        chessboard.addData(i, j, cellData)

        for piece in pieces:
            centroid = piece.getCentroid()
            cell_f32 = np.array(square, dtype=np.float32)
            if cv.pointPolygonTest(cell_f32, centroid, False) > 0:
                cellData = chessboard.getBoardData(i, j)
                if cellData.piece is not None:
                    print("=> Celldata already has a Piece, ignoring")
                    continue
                else:
                    cell.state = CellState.OCCUPIED
                    cellData.piece = piece


chessboard.drawPrediction(orig.copy())


squares_overlay = orig.copy()
im = orig.copy()

chessboard.update()
chessboard.draw(squares_overlay)


im = cv.addWeighted(squares_overlay, 0.5, im, 1 - 0.5, 0)

cv.imwrite("centroids-classes.png", im)
