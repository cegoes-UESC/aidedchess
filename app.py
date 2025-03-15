import cv2 as cv
import numpy as np
from Board import Board, Line
from pathlib import Path
from ultralytics import YOLO
from Perspective import Perspective
import time
from ChessPiece import ChessPiece
from ChessBoard import ChessBoard, ChessBoardCell, ChessBoardData, CellState
from StateManager import stateManager
import os


def scale(boardKeypoints, factor=1.01):

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
        vector = vector[0] * factor, vector[1] * factor
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


model = YOLO("models/final.pt")

images = list(Path("datasets/chess/images/val").glob("*"))

ok = 0
error = 0
errors = []
no_chess = 0
max_it = 0
m_img = ""

start, end = 0, 0
times = []

inf = []

piece_count = 0
piece_error = 0

piece_time = []


piece_det_start, piece_det_end = 0, 0
for image in images:

    start = time.time_ns()

    print("-> Reading image...")

    # image = Path("datasets/chess/images/train/IMG_3394.JPG")

    prediction = model.predict(image.resolve(), verbose=False, save=False)[0]
    inf.append(
        prediction.speed["preprocess"]
        + prediction.speed["inference"]
        + prediction.speed["postprocess"]
    )

    classes, keypoints = prediction.boxes.cls, prediction.keypoints.xy

    boardKeypoints = None

    for c, k in zip(classes, keypoints):
        if c.item() == 0:
            boardKeypoints = np.float32(k.tolist())
            break

    if boardKeypoints is None:
        print("\t-> No chessboard detected")
        no_chess += 1
        continue

    im = cv.imread(str(image.resolve()))
    orig = im.copy()
    squares_overlay = im.copy()

    it = 0

    while True:

        it += 1

        perspective = Perspective(im, boardKeypoints)
        boardPerspective = perspective.apply()

        board = Board(debug=False)
        board.setImage(boardPerspective)
        board.setImageName(image.name)

        try:
            boardResized, centers, (horizontal, vertical), squares = board.process()
            if len(horizontal) == 9 and len(vertical) == 9:
                end = time.time_ns()
                times.append(end - start)
                print("\t-> Ok")
                ok += 1
                if it > max_it:
                    max_it = it
                    m_img = image.resolve()

                piece_det_start = time.time_ns()
                pieces, chessboard = [], []

                for c, k in zip(classes, keypoints):

                    if c.item() == 0:
                        continue

                    piece = ChessPiece()
                    piece.setClass(int(c.item()))

                    for idx, i in enumerate(k):
                        piece.addPoint(i.tolist())

                    pieces.append(piece)
                chessboard = ChessBoard(boardKeypoints)
                boardSquares = perspective.undoPerspective(np.array(squares))
                boardSquares = np.array(boardSquares)
                boardSquares = boardSquares.reshape((8, 8, 4, 2))
                piece_count += len(pieces)
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
                                    piece_error += 1
                                    # print("=> Celldata already has a Piece, ignoring")
                                    continue
                                else:
                                    cell.state = CellState.OCCUPIED
                                    cellData.piece = piece
                piece_det_end = time.time_ns()
                piece_time.append(piece_det_end - piece_det_start)

                n = image.name.split(".")[0]
                os.mkdir(f"results_images/process/{n}")

                for i in range(8):
                    for j in range(8):
                        im_draw = im.copy()
                        cur_pos = stateManager.getState("current_position")
                        cur_pos = (i, j)
                        stateManager.state["current_position"] = cur_pos
                        chessboard.update()
                        chessboard.draw(im_draw)
                        overlay = cv.addWeighted(im_draw, 0.4, im, 0.6, 0)

                        cv.imwrite(f"results_images/process/{n}/{i}_{j}.jpg", overlay)
                break
            else:
                boardKeypoints = scale(boardKeypoints)
                print("\t-> No grid is possible, scaling")
                if it == 20:
                    end = time.time_ns()
                    times.append(end - start)
                    raise Exception("END")
        except Exception:
            print("\t-> Error in process")
            error += 1
            errors.append(image.resolve())
            end = time.time_ns()
            times.append(end - start)
            break


print("OK:", ok, "Total:", len(images))
print("No Chess:", no_chess)
print("Error", error)
print((ok / len(images)) * 100)

print("Max iterations for scale:", max_it)
print(m_img)

times = np.array(times)
inf = np.array(inf)

print(times.max() / 1000000000, times.min() / 1000000000, times.mean() / 1000000000)

print("Inference:", inf.mean())


p_time = np.array(piece_time)
print("Piece time", p_time.mean() / 1000000000)

print("P total", piece_count)
print("P error", piece_error)
print("P Error %", piece_error / piece_count)

print("---")

# j = map(lambda x: str(x.resolve()), errors)

# print("\n".join(j))
