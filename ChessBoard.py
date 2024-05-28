import cv2 as cv
import numpy as np
from ChessPiece import ChessPiece


class ChessBoardCell:
    points: list[float]
    color: tuple[int, int, int]

    def __init__(self, points) -> None:
        self.points = points
        self.color = (125, 10, 71)

    def draw(self, image) -> None:

        cv.fillConvexPoly(
            image,
            np.array(
                [
                    [
                        [self.points[0][0], self.points[0][1]],
                        [self.points[1][0], self.points[1][1]],
                        [self.points[2][0], self.points[2][1]],
                        [self.points[3][0], self.points[3][1]],
                    ]
                ],
                dtype=np.int32,
            ),
            self.color,
        )

    def setColor(self, color: tuple[int, int, int]) -> None:
        self.color = color


class ChessBoardData:
    cell: ChessBoardCell
    piece: ChessPiece | None = None

    def __init__(self, cell: ChessBoardCell, piece: ChessPiece | None = None) -> None:
        self.cell = cell
        self.piece = piece


class ChessBoard:

    data: list[list[ChessBoardData]]

    def __init__(self) -> None:
        arr = np.zeros((8, 8))
        self.data = arr.tolist()

    def addData(self, i, j, data):
        self.data[i][j] = data

    def draw(self, image) -> None:

        for l in self.data:
            for r in l:
                r.cell.draw(image)
