import os
import numpy as np
from game_engine import Minesweeper, game_mode
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

mode = "test"
N = game_mode[mode]["rows"]
M = game_mode[mode]["columns"]

# Constants
NUM_GAMES = 512
INPUT_SHAPE = (N, M)  # Replace with actual dimensions
LABEL_SHAPE = (N, M)
TEST_RATIO = 0.2
VAL_RATIO = 0.2
NUM_CORES = 12  # Number of parallel processes

# Directory structure
DATASET_DIR = "minesweeper_dataset_test_512"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Ensure directories exist
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def generate_game_data(game_id):
    """Generates data for a single game."""
    board = Minesweeper(mode)
    inputs, labels = [], []

    while not (board.game_won or board.game_over):
        board.random_safe_reveal()
        inputs.append(board.get_input())
        labels.append(board.get_output())

    print(f"Game {game_id + 1} completed.")
    return np.array(inputs), np.array(labels)


def save_data_to_file(data, labels, directory, file_prefix):
    """Saves input and label arrays to .npz files."""
    for i, (inp, lbl) in enumerate(zip(data, labels)):
        file_path = os.path.join(directory, f"{file_prefix}_{i:04d}.npz")
        np.savez(file_path, input=inp, label=lbl)


if __name__ == "__main__":
    # Generate data in parallel using joblib
    results = Parallel(n_jobs=NUM_CORES)(
        delayed(generate_game_data)(game_id) for game_id in range(NUM_GAMES)
    )

    # Split and flatten data
    all_inputs = np.concatenate([res[0] for res in results], axis=0)
    all_labels = np.concatenate([res[1] for res in results], axis=0)

    # Split data into train, validation, and test sets
    train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(
        all_inputs, all_labels, test_size=TEST_RATIO + VAL_RATIO, random_state=42
    )
    val_inputs, test_inputs, val_labels, test_labels = train_test_split(
        temp_inputs,
        temp_labels,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        random_state=42,
    )

    # Save datasets
    save_data_to_file(train_inputs, train_labels, TRAIN_DIR, "train")
    save_data_to_file(val_inputs, val_labels, VAL_DIR, "val")
    save_data_to_file(test_inputs, test_labels, TEST_DIR, "test")

    print(f"Dataset generated and saved to {DATASET_DIR}")

# import os
# import numpy as np
# from game_engine import Minesweeper, game_mode
# from sklearn.model_selection import train_test_split

# mode = "intermediate"
# N = game_mode[mode]["rows"]
# M = game_mode[mode]["columns"]
# # Constants
# NUM_GAMES = 8192
# INPUT_SHAPE = (N, M)  # Replace with actual dimensions
# LABEL_SHAPE = (N, M)
# TEST_RATIO = 0.2
# VAL_RATIO = 0.2

# # Directory structure
# DATASET_DIR = "minesweeper_dataset"
# TRAIN_DIR = os.path.join(DATASET_DIR, "train")
# VAL_DIR = os.path.join(DATASET_DIR, "val")
# TEST_DIR = os.path.join(DATASET_DIR, "test")

# # Ensure directories exist
# for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
#     os.makedirs(dir_path, exist_ok=True)


# def generate_game_data():
#     """Generates data for a single game."""
#     board = Minesweeper(mode)
#     inputs, labels = [], []

#     while not (board.game_won or board.game_over):
#         board.random_safe_reveal()
#         inputs.append(board.get_input())
#         labels.append(board.get_output())

#     return np.array(inputs), np.array(labels)


# def save_data_to_file(data, labels, directory, file_prefix):
#     """Saves input and label arrays to .npz files."""
#     for i, (inp, lbl) in enumerate(zip(data, labels)):
#         file_path = os.path.join(directory, f"{file_prefix}_{i:04d}.npz")
#         np.savez(file_path, input=inp, label=lbl)


# if __name__ == "__main__":
#     all_inputs, all_labels = [], []

#     # Generate data for 500 games
#     for game_id in range(NUM_GAMES):
#         print(f"Generating game {game_id + 1}/{NUM_GAMES}")
#         inputs, labels = generate_game_data()
#         all_inputs.append(inputs)
#         all_labels.append(labels)

#     # Flatten inputs and labels for splitting
#     all_inputs = np.concatenate(all_inputs, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)

#     # Split data into train, validation, and test sets
#     train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(
#         all_inputs, all_labels, test_size=TEST_RATIO + VAL_RATIO, random_state=42
#     )
#     val_inputs, test_inputs, val_labels, test_labels = train_test_split(
#         temp_inputs,
#         temp_labels,
#         test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
#         random_state=42,
#     )

#     # Save datasets
#     save_data_to_file(train_inputs, train_labels, TRAIN_DIR, "train")
#     save_data_to_file(val_inputs, val_labels, VAL_DIR, "val")
#     save_data_to_file(test_inputs, test_labels, TEST_DIR, "test")

#     print(f"Dataset generated and saved to {DATASET_DIR}")
