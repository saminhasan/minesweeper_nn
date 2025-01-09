import random
import itertools
import string
import numpy as np
from solver import Rule, MineCount, solve
from enum import Enum

# from scipy.signal import convolve2d


from typing import Any, Set, Dict, List, Tuple

# Type-hinted dictionary for game modes
game_mode: Dict[str, Dict[str, int]] = {
    "test": {"rows": 5, "columns": 5, "mines": 5},
    "easy": {"rows": 10, "columns": 10, "mines": 10},
    "intermediate": {"rows": 16, "columns": 16, "mines": 40},
    "hard": {"rows": 16, "columns": 40, "mines": 99},
    "xtreme": {"rows": 20, "columns": 40, "mines": 128},
}


# State enum for cell states
class State(Enum):
    UNCOVERED = 0
    COVERED = -1
    FLAGGED = 1


class TagGenerator:
    def __init__(self) -> None:
        self.chars: str = string.ascii_uppercase
        self.current: int = 0
        self.tag_length: int = 1

    def next_tag(self) -> str:
        if self.current >= len(self.chars) ** self.tag_length:
            self.tag_length += 1
            self.current = 0

        indices: List[int] = []
        temp: int = self.current
        for _ in range(self.tag_length):
            indices.append(temp % len(self.chars))
            temp //= len(self.chars)

        tag: str = "".join(self.chars[i] for i in reversed(indices))
        self.current += 1
        return tag


class Minesweeper:
    def __init__(self, difficulty: str) -> None:
        self.game_over: bool = False
        self.game_won: bool = False
        self.states: Enum = State
        self.n_rows: int = game_mode[difficulty]["rows"]
        self.n_cols: int = game_mode[difficulty]["columns"]
        self.shape: Tuple[int, int] = (self.n_rows, self.n_cols)
        self.n_mines: int = game_mode[difficulty]["mines"]
        self.cell_dtype: np.dtype = np.dtype(
            [
                ("mine_count", np.int8),
                ("state", np.int8),
            ]
        )
        self.minefield: np.ndarray = np.zeros((self.n_rows, self.n_cols), dtype=self.cell_dtype)
        self.minefield["state"] = self.states.COVERED.value
        self.mines: Set[Tuple[int, int]] = set()
        self.place_mines()

    def place_mines(self) -> None:
        indices: List[Tuple[int, int]] = list(np.indices(self.shape).reshape(len(self.shape), -1).T)
        mine_indices: List[Tuple[int, int]] = random.sample(indices, self.n_mines)
        for i, j in mine_indices:
            self.mines.add((i, j))
            self.minefield[i, j]["mine_count"] = -1
            row_range = slice(max(0, i - 1), min(i + 2, self.n_rows))
            col_range = slice(max(0, j - 1), min(j + 2, self.n_cols))
            neighbors = self.minefield[row_range, col_range]
            no_mine = neighbors["mine_count"] != -1
            neighbors["mine_count"][no_mine] += 1

    def reveal(self, i: int, j: int) -> None:
        if self.game_over or self.game_won or self.minefield[i, j]["state"] != State.COVERED.value:
            return

        if self.minefield[i, j]["mine_count"] == -1:
            self.game_over = True
            self.reveal_all_mines()
            print("Game Over! You hit a mine.")
            return

        def flood_fill(row: int, col: int):
            stack = [(row, col)]
            while stack:
                x, y = stack.pop()
                if self.minefield[x, y]["state"] == State.COVERED.value:
                    self.minefield[x, y]["state"] = State.UNCOVERED.value
                    if self.minefield[x, y]["mine_count"] == 0:
                        for nx in range(max(0, x - 1), min(x + 2, self.n_rows)):
                            for ny in range(max(0, y - 1), min(y + 2, self.n_cols)):
                                if (nx, ny) != (x, y):
                                    stack.append((nx, ny))

        flood_fill(i, j)

        if self.check_win():
            self.game_won = True
            print("You won!", flush=True)

    def random_safe_reveal(self) -> None:
        covered_mask = (self.minefield["state"] == State.COVERED.value) & (self.minefield["mine_count"] != -1)
        safe_cells = np.argwhere(covered_mask)

        if safe_cells.size == 0:
            print("No safe cells to reveal.")
            return

        i, j = safe_cells[np.random.choice(safe_cells.shape[0])]
        self.reveal(i, j)

    def reveal_all_mines(self) -> None:
        mine_coords = np.array(list(self.mines))
        self.minefield[mine_coords[:, 0], mine_coords[:, 1]]["state"] = State.UNCOVERED.value
        self.game_over = True
        self.game_won = False

    def check_win(self) -> bool:
        covered_cells = np.sum(self.minefield["state"] == State.COVERED.value)
        return covered_cells == self.n_mines

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        row_range = slice(max(0, i - 1), min(i + 2, self.n_rows))
        col_range = slice(max(0, j - 1), min(j + 2, self.n_cols))
        neighbors = [
            (x, y)
            for x in range(row_range.start, row_range.stop)
            for y in range(col_range.start, col_range.stop)
            if (x, y) != (i, j)
        ]
        return neighbors

    # def get_frontier_cells(self) -> np.ndarray:
    #     """
    #     Computes the frontier cells that are adjacent to revealed cells.

    #     Returns:
    #         np.ndarray: A binary matrix indicating frontier cells (1 for frontier, 0 otherwise).
    #     """
    #     # Create a matrix to represent the revealed state of each cell
    #     revealed_matrix: np.ndarray = np.array(
    #         [
    #             [cell["state"] == State.UNCOVERED.value for cell in row]
    #             for row in self.minefield
    #         ]
    #     )

    #     # Define a 3x3 kernel filled with ones
    #     kernel: np.ndarray = np.ones((3, 3))

    #     # Perform 2D convolution to count the number of revealed neighbors for each cell
    #     convolved: np.ndarray = convolve2d(revealed_matrix, kernel, mode="same")

    #     # Identify cells that are not revealed but are adjacent to at least one revealed cell
    #     frontier: np.ndarray = np.logical_and(
    #         convolved > 0, revealed_matrix == 0
    #     ).astype(int)
    #     return frontier

    def create_rules_from_minefield(self) -> List[Any]:
        rules: List[Any] = []
        tags: Dict[Tuple[int, int], str] = {}
        tag_generator: TagGenerator = TagGenerator()
        self.tag_to_index: Dict[str, Tuple[int, int]] = {}

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.minefield[i, j]["state"] == State.UNCOVERED.value:
                    neighbors = self.get_neighbors(i, j)
                    mine_count: int = self.minefield[i, j]["mine_count"]
                    covered_neighbors: List[str] = []

                    for x, y in neighbors:
                        if self.minefield[x, y]["state"] == State.COVERED.value or State.FLAGGED.value:
                            if (x, y) not in tags:
                                tag: str = tag_generator.next_tag()
                                tags[(x, y)] = tag
                                self.tag_to_index[tag] = (x, y)
                            covered_neighbors.append(tags[(x, y)])

                    if covered_neighbors:
                        rules.append(Rule(mine_count, covered_neighbors))

        return rules

    def decode_solution(self, solution: Dict[str, float]) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
        decoded_solution: Dict[Tuple[int, int], float] = {}
        try:
            probability_array: np.ndarray = np.zeros((self.n_rows, self.n_cols)) + solution[None]
        except KeyError:
            probability_array = np.zeros((self.n_rows, self.n_cols))

        for tag, probability in solution.items():
            if tag in self.tag_to_index:
                decoded_solution[self.tag_to_index[tag]] = probability
                probability_array[self.tag_to_index[tag]] = probability

        return decoded_solution, probability_array

    def solve_minefield(self) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
        rules: List[Any] = self.create_rules_from_minefield()
        total_cells: int = self.n_rows * self.n_cols
        results = solve(rules, MineCount(total_cells=total_cells, total_mines=self.n_mines))
        return self.decode_solution(results)

    def get_input(self) -> np.ndarray:
        display_array: np.ndarray = np.full_like(self.minefield["mine_count"], self.states.COVERED.value)
        display_array[self.minefield["state"] == self.states.UNCOVERED.value] = self.minefield["mine_count"][
            self.minefield["state"] == self.states.UNCOVERED.value
        ]
        display_array = ((display_array + 1) / 9) * (display_array >= 0)
        return display_array

    def get_output(self) -> np.ndarray:
        _, probability = self.solve_minefield()
        probability[self.minefield["state"] == self.states.UNCOVERED.value] = 0
        return probability

    @staticmethod
    def display_minefield(minefield: np.ndarray) -> None:
        rows, cols = minefield.shape
        for i in range(rows):
            for j in range(cols):
                cell = minefield[i, j]
                if cell["state"] == State.COVERED.value:
                    print("#", end=" ")
                elif cell["state"] == State.FLAGGED.value:
                    print("F", end=" ")
                elif cell["state"] == State.UNCOVERED.value:
                    if cell["mine_count"] == -1:
                        print("*", end=" ")  # Print an asterisk for mines
                    else:
                        print(
                            cell["mine_count"] if cell["mine_count"] >= 0 else " ",
                            end=" ",
                        )
            print()  # Newline after each row


import time
from statistics import mean
from joblib import Parallel, delayed


def play_game(game_id):
    """
    Function to simulate a single Minesweeper game.
    """
    start_time = time.time()
    board: Minesweeper = Minesweeper("easy")
    counter: int = 0

    # Play the game until it's won or lost
    while not (board.game_won or board.game_over):
        board.random_safe_reveal()
        a, b = board.solve_minefield()
        counter += 1

    # Record time taken and number of moves
    time_taken = time.time() - start_time
    return counter, time_taken, game_id


if __name__ == "__main__":
    total_games = 1000
    n_jobs = 12  # Number of parallel cores to use

    print(f"Simulating {total_games} games using {n_jobs} cores...\n")

    # Run games in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(play_game)(game_id) for game_id in range(total_games))

    # Extract moves and time for each game
    moves_list = [result[0] for result in results]
    time_list = [result[1] for result in results]

    # Print game details (optional)
    for counter, (moves, time_taken, game_id) in enumerate(results):
        print(f"Game: {game_id + 1}, Moves: {moves}, Time Taken: {time_taken:.2f} seconds.")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Games: {total_games}")
    print(f"Moves: Min={min(moves_list)}, Max={max(moves_list)}, Avg={mean(moves_list):.2f}")
    print(f"Time: Min={min(time_list):.2f} seconds, Max={max(time_list):.2f} seconds, Avg={mean(time_list):.2f} seconds")
