from solver import * 

print(solve([
    Rule(1, ['A', 'B']),
    Rule(2, ['A', 'B', 'C']),
    Rule(3, ['B', 'C', 'D']),
    Rule(2, ['C', 'D', 'E']),
    Rule(2, ['D', 'E', 'F', 'G', 'H']),
    Rule(1, ['G', 'H', 'I']),
    Rule(1, ['H', 'I']),
], MineCount(total_cells=85, total_mines=10)))