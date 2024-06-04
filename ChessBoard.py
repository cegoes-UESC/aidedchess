import cv2 as cv
import numpy as np
from ChessPiece import ChessPiece, PieceType
from StateManager import stateManager


class CellState:
    EMPTY = 0
    OCCUPIED = 1
    SELECTED = 2
    CAPTURABLE = 3
    MOVEABLE = 4


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
        elif self.state == CellState.MOVEABLE:
            color = (255, 255, 0)

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

    def goUp(self):
        cur_pos = stateManager.getState("current_position")
        if cur_pos[0] > 0:
            cur_pos[0] -= 1
            stateManager["current_position"] = cur_pos

    def goDown(self):
        cur_pos = stateManager.getState("current_position")
        if cur_pos[0] < 7:
            cur_pos[0] += 1
            stateManager["current_position"] = cur_pos

    def goLeft(self):
        cur_pos = stateManager.getState("current_position")
        if cur_pos[1] > 0:
            cur_pos[1] -= 1
            stateManager["current_position"] = cur_pos

    def goRight(self):
        cur_pos = stateManager.getState("current_position")
        if cur_pos[1] < 7:
            cur_pos[1] += 1
            stateManager["current_position"] = cur_pos

    def clear(self):
        for i in self.data:
            for data in i:
                data.cell.setState(
                    CellState.EMPTY if data.piece is None else CellState.OCCUPIED
                )

    def handleSelected(self):
        pos = stateManager.getState("current_position")
        selectedCell: ChessBoardData = self.data[pos[0]][pos[1]]
        selectedCell.cell.setState(CellState.SELECTED)

        if selectedCell.piece is not None:
            if selectedCell.piece.type == PieceType.QUEEN:
                self.checkDiagonalMove(pos, selectedCell)
                self.checkSideMove(pos, selectedCell)
            elif selectedCell.piece.type == PieceType.BISHOP:
                self.checkDiagonalMove(pos, selectedCell)
            elif selectedCell.piece.type == PieceType.ROOK:
                self.checkSideMove(pos, selectedCell)

    def update(self):
        self.clear()
        self.handleSelected()
        pos = stateManager.getState("current_position")
        print(pos)

    def checkDiagonalMove(self, pos: tuple[int, int], selectedCell: ChessBoardData):
        p1, p2 = pos[0] + 1, pos[1] + 1
        while p1 < 8 and p2 < 8:
            aux = self.data[p1][p2]
            if aux.piece is not None:
                if aux.piece.color == selectedCell.piece.color:
                    break
                else:
                    aux.cell.setState(CellState.CAPTURABLE)
                    break
            else:
                aux.cell.setState(CellState.MOVEABLE)
            p1 += 1
            p2 += 1

        p1, p2 = pos[0] - 1, pos[1] - 1
        while p1 > -1 and p2 > -1:
            aux = self.data[p1][p2]
            if aux.piece is not None:
                if aux.piece.color == selectedCell.piece.color:
                    break
                else:
                    aux.cell.setState(CellState.CAPTURABLE)
                    break
            else:
                aux.cell.setState(CellState.MOVEABLE)
            p1 -= 1
            p2 -= 1

        p1, p2 = pos[0] + 1, pos[1] - 1
        while p1 < 8 and p2 > -1:
            aux = self.data[p1][p2]
            if aux.piece is not None:
                if aux.piece.color == selectedCell.piece.color:
                    break
                else:
                    aux.cell.setState(CellState.CAPTURABLE)
                    break
            else:
                aux.cell.setState(CellState.MOVEABLE)
            p1 += 1
            p2 -= 1

        p1, p2 = pos[0] - 1, pos[1] + 1
        while p1 > -1 and p2 < 8:
            aux = self.data[p1][p2]
            if aux.piece is not None:
                if aux.piece.color == selectedCell.piece.color:
                    break
                else:
                    aux.cell.setState(CellState.CAPTURABLE)
                    break
            else:
                aux.cell.setState(CellState.MOVEABLE)
            p1 -= 1
            p2 += 1

    def checkSideMove(self, pos: tuple[int, int], selectedCell: ChessBoardData):
        if pos[0] + 1 < 7:
            for i in range(pos[0] + 1, 8, 1):
                aux = self.data[i][pos[1]]
                if aux.piece is not None:
                    if aux.piece.color == selectedCell.piece.color:
                        break
                    else:
                        aux.cell.setState(CellState.CAPTURABLE)
                        break
                else:
                    aux.cell.setState(CellState.MOVEABLE)
        if pos[0] - 1 > 0:
            for i in range(pos[0] - 1, -1, -1):
                aux = self.data[i][pos[1]]
                if aux.piece is not None:
                    if aux.piece.color == selectedCell.piece.color:
                        break
                    else:
                        aux.cell.setState(CellState.CAPTURABLE)
                        break
                else:
                    aux.cell.setState(CellState.MOVEABLE)

        if pos[1] + 1 < 7:
            for i in range(pos[1] + 1, 8, 1):
                aux = self.data[pos[0]][i]
                if aux.piece is not None:
                    if aux.piece.color == selectedCell.piece.color:
                        break
                    else:
                        aux.cell.setState(CellState.CAPTURABLE)
                        break
                else:
                    aux.cell.setState(CellState.MOVEABLE)

        if pos[1] - 1 > 0:
            for i in range(pos[1] - 1, -1, -1):
                aux = self.data[pos[0]][i]
                if aux.piece is not None:
                    if aux.piece.color == selectedCell.piece.color:
                        break
                    else:
                        aux.cell.setState(CellState.CAPTURABLE)
                        break
                else:
                    aux.cell.setState(CellState.MOVEABLE)
