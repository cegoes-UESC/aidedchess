from enum import Enum, IntEnum


class PieceType(str, Enum):
    PAWN = "pawn"
    ROOK = "rook"
    BISHOP = "bishop"
    KNIGHT = "knight"
    QUEEN = "queen"
    KING = "king"


class PieceColor(str, Enum):
    BLACK = "black"
    WHITE = "white"


class PieceClass(IntEnum):
    BLACK_PAWN = 1
    WHITE_PAWN = 2
    BLACK_ROOK = 3
    WHITE_ROOK = 4
    BLACK_BISHOP = 5
    WHITE_BISHOP = 6
    BLACK_KNIGHT = 7
    WHITE_KNIGHT = 8
    BLACK_QUEEN = 9
    WHITE_QUEEN = 10
    BLACK_KING = 11
    WHITE_KING = 12


class ChessPiece:

    cls: PieceClass
    points: list[tuple[int, int]]

    def __init__(self) -> None:
        self.cls = 0
        self.points = []

    def getCentroid(self) -> float:
        x, y = 0, 0
        for idx, i in enumerate(self.points):
            if idx == 0:
                continue
            x = x + i[0]
            y = y + i[1]
        return int(1 / 3 * x), int(1 / 3 * y)

    def addPoint(self, point: tuple[int, int]) -> None:
        self.points.append(point)

    def setClass(self, cls: PieceClass) -> None:
        self.cls = cls
