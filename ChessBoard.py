import cv2 as cv
import numpy as np
from ChessPiece import ChessPiece
from typing import List, Tuple, Optional


class ChessBoardCell:
    points: List[float]
    color: Tuple[int, int, int]

    def __init__(self) -> None:
        self.points = []
        self.color = (0, 0, 0)

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

    def setColor(self, color: Tuple[int, int, int]) -> None:
        self.color = color


class ChessBoardData:
    cell: ChessBoardCell
    piece: Optional[ChessPiece] = None

    def __init__(self) -> None:
        self.piece = None
        self.cell = None


class ChessBoard:

    data: List[List[ChessBoardData]]

    def __init__(self) -> None:
        self.data = [[]]
