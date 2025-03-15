from enum import Enum, IntEnum
import cv2 as cv


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
    type: PieceType = None
    color: PieceColor = None
    points: list[tuple[int, int]]
    x: int = 0
    y: int = 0
    showKeypoints: bool = False

    def __init__(self) -> None:
        self.cls = -1
        self.points = []

    def getCentroid(self) -> float:
        x, y = 0, 0
        for idx, i in enumerate(self.points):
            if idx == 0:
                continue
            x = x + i[0]
            y = y + i[1]
        x, y = int(1 / 3 * x), int(1 / 3 * y)
        self.x = x
        self.y = y
        return x, y

    def addPoint(self, point: tuple[int, int]) -> None:
        self.points.append(point)

    def setClass(self, cls: PieceClass) -> None:
        self.cls = cls
        type, color = self._getTypeAndColorFromClass()
        self.setType(type)
        self.setColor(color)

    def setType(self, type: PieceType) -> None:
        self.type = type

    def setColor(self, color: PieceColor) -> None:
        self.color = color

    def setShowKeypoints(self, show=False) -> None:
        self.showKeypoints = show

    def _getTypeAndColorFromClass(self) -> tuple[PieceType, PieceColor]:
        type, color = None, None

        color = PieceColor.WHITE if self.cls % 2 == 0 else PieceColor.BLACK

        if self.cls in [1, 2]:
            type = PieceType.PAWN
        elif self.cls in [3, 4]:
            type = PieceType.ROOK
        elif self.cls in [5, 6]:
            type = PieceType.BISHOP
        elif self.cls in [7, 8]:
            type = PieceType.KNIGHT
        elif self.cls in [9, 10]:
            type = PieceType.QUEEN
        elif self.cls in [11, 12]:
            type = PieceType.KING

        return type, color

    def __str__(self) -> str:
        if self.color and self.type:
            return f"A {self.color.value} {self.type.value} at ({self.x}, {self.y})"
        return "Empty Piece"

    def draw(self, image) -> None:

        color = (0, 255, 255) if self.color == PieceColor.BLACK else (0, 255, 255)

        cv.drawMarker(
            image,
            (int(self.x), int(self.y)),
            color,
            cv.MARKER_DIAMOND,
            20,
            15,
        )

        if self.showKeypoints:
            for p in self.points:
                cv.drawMarker(
                    image,
                    (int(p[0]), int(p[1])),
                    (255, 120, 70),
                    cv.MARKER_DIAMOND,
                    15,
                    10,
                )
