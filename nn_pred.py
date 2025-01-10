import os
import ctypes
import numpy as np
import pygame as pg
import pygame.gfxdraw
from game_engine import Minesweeper
from render_util import (
    blur_bg,
    find_clusters,
    get_custom_rgb,
    draw_polygon_with_holes,
    rects_to_polygon,
    get_rects_from_cluster,
)
from typing import Dict, Tuple, Callable, Set
from tensorflow.keras.models import load_model  # type: ignore

# Load the model
model = load_model("md.h5")
model.summary()


def nn_pred(board: Minesweeper):
    return model.predict(np.expand_dims(np.expand_dims(board.get_input(), axis=0), axis=-1))[0, :, :, 0]


FONT_PATH: str = "./assets/fonts"
IMAGE_PATH: str = "./assets/images"
CELL_SIZE: int = 64
LINE_WIDTH: int = 0
BORDER_SIZE: int = 0
COVERED: int = -1
UNCOVERED: int = 0

cell_color: Tuple[int, int, int] = (30, 30, 30)
line_color: Tuple[int, int, int] = (125, 125, 125)
background_color: Tuple[int, int, int] = (5, 5, 5)
text_color: Tuple[int, int, int] = (220, 220, 220)
font_size: int = 16

levels = {0: "test", 1: "easy", 2: "intermediate", 3: "hard", 4: "xtreme"}


class GUI:
    def __init__(self, level: str):
        self.level = levels[level]  # Store the level for resetting the game
        self.init = True
        self.init_game()

    def init_game(self):
        if self.init:
            pg.quit()

        self.help = True
        """
        Initialize the Minesweeper GUI.

        Parameters:
            level (str): The difficulty level of the Minesweeper game (e.g., 'easy', 'medium', 'hard').
        """
        self.board = Minesweeper(self.level)  # Create a Minesweeper board
        # Solve the minefield and get the solution and probabilities
        # _, self.probability = self.board.solve_minefield()
        self.probability = nn_pred(self.board)

        # Calculate board dimensions
        rows, cols = self.board.shape
        self.width: int = BORDER_SIZE * 2 + cols * CELL_SIZE + (cols - 1) * (LINE_WIDTH + BORDER_SIZE * 2) + 1
        self.height: int = BORDER_SIZE * 2 + rows * CELL_SIZE + (rows - 1) * (LINE_WIDTH + BORDER_SIZE * 2) + 1

        # Initialize game state
        self.running: bool = True
        self.fps: int = 240
        # Initialize pygame
        pg.init()
        pg.font.init()
        self.mine_image: pg.Surface = pg.image.load(f"{IMAGE_PATH}/mine.png")
        # Scale the mine image to fit the cell size
        self.scaled_mine_image = pg.transform.scale(self.mine_image, (CELL_SIZE, CELL_SIZE))
        self.flag_image = pg.image.load(f"{IMAGE_PATH}/flag.png")
        self.scaled_flag_image = pg.transform.scale(self.flag_image, (CELL_SIZE // 2, CELL_SIZE // 2))

        pg.display.set_icon(self.mine_image)  # Set the icon for the window

        self.font: pg.font.Font = pg.font.Font(f"{FONT_PATH}/orbitron/orbitron.ttf", font_size)

        self.clock: pg.time.Clock = pg.time.Clock()

        # # Calculate the screen's center position for the window
        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{(screen_width - self.width) // 2},{(screen_height - self.height) // 2}"
        self.screen: pg.Surface = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Minesweeper")

        self.flagged: Set[Tuple[int, int]] = set()

    def quit(self) -> None:
        """
        Stop the game and exit the main loop.
        """
        self.running = False

    def reset_game(self) -> None:
        """
        Reset the game by reinitializing the Minesweeper board.
        """
        self.init_game()

    def handle_events(self) -> None:
        """
        Handle all pygame events, including quit, key presses, and mouse actions.
        """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            elif event.type == pg.KEYDOWN:
                self.handle_key_event(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.handle_mouse_event(event)

    def handle_key_event(self, key: int) -> None:
        """
        Handle key events for the game.

        Parameters:
            key (int): The key code of the pressed key.
        """
        if key == pg.K_ESCAPE:
            self.quit()
        elif key == pg.K_r:
            self.reset_game()
        elif key == pg.K_h:
            self.help = not self.help
            if self.help:
                self.probability = nn_pred(self.board)
        else:
            pass

    def handle_mouse_event(self, event: pg.event.Event) -> None:
        """
        Handle mouse events for the game.

        Parameters:
            event (pg.event.Event): The pygame mouse event object.
        """
        if not (self.board.game_over or self.board.game_won):
            if event.type == pg.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos  # Get mouse position
                col = (mouse_x - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
                row = (mouse_y - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
                if 0 <= row < self.board.n_rows and 0 <= col < self.board.n_cols:
                    if event.button == pg.BUTTON_LEFT:
                        if not self.board.game_over:
                            if (row, col) not in self.flagged:
                                self.board.reveal(row, col)
                                self.probability = nn_pred(self.board)
                                if not self.board.game_over:
                                    self.flagged = {
                                        flag
                                        for flag in self.flagged
                                        if self.board.minefield[flag]["state"] != self.board.states.UNCOVERED
                                    }
                    if event.button == pg.BUTTON_RIGHT:
                        if (row, col) not in self.flagged and self.board.minefield[row, col][
                            "state"
                        ] != self.board.states.UNCOVERED:
                            self.flagged.add((row, col))
                        else:
                            self.flagged.discard((row, col))

    def draw(self) -> None:
        """
        Render the Minesweeper game board on the screen.
        """
        # Fill the screen with the background color
        self.screen.fill(background_color)

        # Check for game state
        if self.board.game_over:
            # Handle game over
            self.draw_mines()
            # Create a transparent overlay surface
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((128, 0, 0, 64))  # Semi-transparent red background
            self.screen.blit(overlay, (0, 0))
            # Display game over message
            main_text = "Game Over, Press 'R' to Restart"
            self.draw_lines()
        elif self.board.game_won:
            # Handle game won
            self.draw_clusters()
            self.draw_flags()

            # Create a transparent overlay surface
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 128, 0, 64))  # Semi-transparent green background
            self.screen.blit(overlay, (0, 0))

            # Display game won message
            main_text = "Game Won, Press 'R' to Restart"
            self.draw_lines()
        else:
            # Handle normal game state
            self.draw_clusters()
            self.draw_cells_bayes()
            self.draw_lines()
            self.draw_markers()
            blur_bg(self.screen, sigma=0.25)

            return  # Skip the rest since no overlay or text is needed
        blur_bg(self.screen, sigma=2)
        # Combine all text rendering into one loop
        # Define all text content
        text_lines = [
            main_text,
            f"Current Level: {self.level.capitalize()}",  # Level text
            "Press H in game to Toggle Help",
        ]

        # Calculate the vertical spacing dynamically
        total_texts = len(text_lines)
        start_y = self.height // 4  # Start at 1/4th of the screen height
        end_y = (3 * self.height) // 4  # End at 3/4th of the screen height
        vertical_spacing = (end_y - start_y) // (total_texts - 1)  # Evenly space the text

        # Render and position all text
        for i, text in enumerate(text_lines):
            # Render the text
            text_surface = self.font.render(text, True, (255, 255, 255))  # White text
            # Calculate the text rectangle (centered horizontally, adjusted vertically)
            text_rect = text_surface.get_rect(center=(self.width // 2, start_y + i * vertical_spacing))
            # Draw the text
            self.screen.blit(text_surface, text_rect)

    def draw_clusters(self):
        [
            draw_polygon_with_holes(
                self.screen,
                rects_to_polygon(get_rects_from_cluster(cluster, CELL_SIZE, BORDER_SIZE, LINE_WIDTH)),
                cell_color,
                background_color,
                CELL_SIZE,
            )
            for cluster in find_clusters(self.board.minefield["state"], COVERED)
        ]
        blur_bg(self.screen, sigma=0.8)

    def draw_mines(self):
        [
            self.screen.blit(
                self.scaled_mine_image,
                (
                    BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1,
                    BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1,
                ),
            )
            for row, col in self.board.mines
        ]

    def draw_flags(self):
        [
            self.screen.blit(
                self.scaled_flag_image,
                (
                    BORDER_SIZE
                    + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH)
                    + 1
                    + self.scaled_flag_image.get_width() // 2,
                    BORDER_SIZE
                    + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH)
                    + 1
                    + self.scaled_flag_image.get_height() // 2,
                ),
            )
            for row, col in self.board.mines
        ]

    def draw_markers(self):

        [
            self.screen.blit(
                self.scaled_flag_image,
                (
                    BORDER_SIZE
                    + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH)
                    + 1
                    + self.scaled_flag_image.get_width() // 2,
                    BORDER_SIZE
                    + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH)
                    + 1
                    + self.scaled_flag_image.get_height() // 2,
                ),
            )
            for row, col in self.flagged
        ]

    def draw_cells_bayes(self):
        # Draw cells with numbers or probabilities
        for row in range(self.board.n_rows):
            for col in range(self.board.n_cols):
                x = BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                y = BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                cell = self.board.minefield[row, col]

                # Draw uncovered cells with mine counts
                if cell["state"] == self.board.states.UNCOVERED:
                    if cell["mine_count"] > 0:
                        text_surface = self.font.render(f"{cell['mine_count']}", True, text_color)
                        text_rect = text_surface.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                        self.screen.blit(text_surface, text_rect)
                if self.help:
                    if (row, col) not in self.flagged:
                        if cell["state"] == self.board.states.COVERED:
                            if self.probability is not None:
                                probability = self.probability[row, col]
                                if probability == 0:
                                    display_text = "S"
                                elif probability == 1:
                                    display_text = "X"
                                else:
                                    display_text = f"{probability:.2f}"
                                text_surface = self.font.render(display_text, True, get_custom_rgb(probability))
                                text_rect = text_surface.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                                self.screen.blit(text_surface, text_rect)

    def draw_lines(self):
        # Grid line parameters
        gap_size = CELL_SIZE // 8
        segment_length = CELL_SIZE - gap_size

        # Draw vertical grid lines
        [
            pg.draw.line(
                self.screen,
                line_color,
                (
                    BORDER_SIZE * 2 * (col + 1) + CELL_SIZE * (col + 1) + LINE_WIDTH * col,
                    seg_start + gap_size,
                ),
                (
                    BORDER_SIZE * 2 * (col + 1) + CELL_SIZE * (col + 1) + LINE_WIDTH * col,
                    min(seg_start + segment_length, self.height),
                ),
                1,
            )
            for col in range(self.board.n_cols - 1)
            for seg_start in range(0, self.height, segment_length + gap_size)
        ]

        # Draw horizontal grid lines
        [
            pg.draw.line(
                self.screen,
                line_color,
                (
                    seg_start + gap_size,
                    BORDER_SIZE * 2 * (row + 1) + CELL_SIZE * (row + 1) + LINE_WIDTH * row,
                ),
                (
                    min(seg_start + segment_length, self.width),
                    BORDER_SIZE * 2 * (row + 1) + CELL_SIZE * (row + 1) + LINE_WIDTH * row,
                ),
                1,
            )
            for row in range(self.board.n_rows - 1)
            for seg_start in range(0, self.width, segment_length + gap_size)
        ]


def main():
    game = GUI(1)

    while game.running:
        game.handle_events()
        game.draw()
        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    main()
