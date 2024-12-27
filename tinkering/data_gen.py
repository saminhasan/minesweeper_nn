import numpy as np
from game_engine import Minesweeper


if __name__ == "__main__":
    board: Minesweeper = Minesweeper("easy")
    counter: int = 0
    while (not (board.game_won or board.game_over)) and counter < 10:
        board.random_safe_reveal()
        print(board.get_input())
        print(board.get_output())
        print(counter)
        break
