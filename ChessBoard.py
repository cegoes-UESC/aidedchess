import cv2 as cv
import numpy as np
from ChessPiece import ChessPiece


class CellState:
    EMPTY = 0
    OCCUPIED = 1
    SELECTED = 2
    CAPTURABLE = 3


class ChessBoardCell:
    points: list[float]
    state: CellState = 0

    def __init__(self, points) -> None:
        self.points = points

    def draw(self, image) -> None:

        color = (255, 255, 255)

        if self.state == CellState.OCCUPIED:
            color = (0, 255, 0)
        elif self.state == CellState.SELECTED:
            color = (255, 0, 0)
        elif self.state == CellState.CAPTURABLE:
            color = (0, 0, 255)

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
            color,
        )

    def setState(self, state: CellState) -> None:
        self.state = state


class ChessBoardData:
    cell: ChessBoardCell
    piece: ChessPiece | None = None

    def __init__(self, cell: ChessBoardCell, piece: ChessPiece | None = None) -> None:
        self.cell = cell
        self.piece = piece


class ChessBoard:

    data: list[list[ChessBoardData]]
    points: list = []

    def __init__(self, points: list) -> None:
        arr = np.zeros((8, 8))
        self.data = arr.tolist()
        self.points = points

    def addData(self, i, j, data):
        self.data[i][j] = data

    def draw(self, image) -> None:

        for l in self.data:
            for r in l:
                if isinstance(r, ChessBoardData):
                    r.cell.draw(image)
                    if r.piece is not None:
                        r.piece.draw(image)

        # cv.fillConvexPoly(
        #     image,
        #     np.array(
        #         [
        #             [
        #                 [self.points[0][0], self.points[0][1]],
        #                 [self.points[1][0], self.points[1][1]],
        #                 [self.points[2][0], self.points[2][1]],
        #                 [self.points[3][0], self.points[3][1]],
        #             ]
        #         ],
        #         dtype=np.int32,
        #     ),
        #     (0, 0, 255),
        # )

    def getBoardData(self, i, j) -> ChessBoardData | None:
        return self.data[i][j] if isinstance(self.data[i][j], ChessBoardData) else None
