import os
import ctypes
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
font_size: int = 15

levels = {0: "test", 1: "easy", 2: "intermediate", 3: "hard", 4: "xtreme"}


def predict(board):
    _, probability = board.solve_minefield()
    return probability


class GUI:
    def __init__(self, level: str):
        self.level: str = levels[level]
        self.init: bool = True
        self.init_game()

    def init_game(self):
        if self.init:
            pg.quit()

        # Init game state
        self.running: bool = True
        self.fps: int = 240
        self.help: bool = True
        self.flagged: Set[Tuple[int, int]] = set()

        # Initboard
        self.board = Minesweeper(self.level)
        rows, cols = self.board.shape
        self.probability = predict(self.board)

        # Dynamically adjust CELL_SIZE based on rows or columns
        global CELL_SIZE
        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        if rows >= cols:
            CELL_SIZE = (screen_height - 100 - BORDER_SIZE * 2 - (rows - 1) * (LINE_WIDTH + BORDER_SIZE * 2)) // rows
        else:
            CELL_SIZE = (screen_width - 100 - BORDER_SIZE * 2 - (cols - 1) * (LINE_WIDTH + BORDER_SIZE * 2)) // cols

        self.width: int = BORDER_SIZE * 2 + cols * CELL_SIZE + (cols - 1) * (LINE_WIDTH + BORDER_SIZE * 2) + 1
        self.height: int = BORDER_SIZE * 2 + rows * CELL_SIZE + (rows - 1) * (LINE_WIDTH + BORDER_SIZE * 2) + 1

        # Set window position to the center of the screen
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{(screen_width - self.width) // 2},{(screen_height - self.height - 20) // 2}"

        # Init pg
        pg.init()
        pg.font.init()
        self.clock: pg.time.Clock = pg.time.Clock()
        self.screen: pg.Surface = pg.display.set_mode((self.width, self.height))

        self.font: pg.font.Font = pg.font.Font(f"{FONT_PATH}/orbitron/orbitron.ttf", font_size)
        self.mine_image: pg.Surface = pg.image.load(f"{IMAGE_PATH}/mine.png").convert_alpha()
        self.flag_image = pg.image.load(f"{IMAGE_PATH}/flag.png").convert_alpha()
        self.scaled_mine_image = pg.transform.scale(self.mine_image, (CELL_SIZE // 2, CELL_SIZE // 2))
        self.scaled_flag_image = pg.transform.scale(self.flag_image, (CELL_SIZE // 2, CELL_SIZE // 2))
        # Configure the game window
        pg.display.set_caption("Minesweeper")
        pg.display.set_icon(self.mine_image)

    def quit(self) -> None:
        self.running = False

    def reset_game(self) -> None:
        self.init_game()

    def handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            elif event.type == pg.KEYDOWN:
                self.handle_key_event(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.handle_mouse_event(event)

    def handle_key_event(self, key: int) -> None:
        if key == pg.K_ESCAPE:
            self.quit()
        elif key == pg.K_r:
            self.reset_game()
        elif key == pg.K_h:
            self.help = not self.help
            if self.help:
                self.probability = predict(self.board)
        elif key in [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4]:
            self.level = levels[key - pg.K_0]
            self.reset_game()
        else:
            pass

    def handle_mouse_event(self, event: pg.event.Event) -> None:
        if not (self.board.game_over or self.board.game_won):
            if event.type == pg.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                col = (mouse_x - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
                row = (mouse_y - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
                if 0 <= row < self.board.n_rows and 0 <= col < self.board.n_cols:
                    if event.button == pg.BUTTON_LEFT:
                        if not self.board.game_over:
                            if (row, col) not in self.flagged:
                                self.board.reveal(row, col)
                                self.probability = predict(self.board)

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
        self.screen.fill(background_color)

        if self.board.game_over:
            self.draw_mines()
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((128, 0, 0, 64))
            self.screen.blit(overlay, (0, 0))
            main_text = "Game Over, Press 'R' to Restart"
        elif self.board.game_won:
            self.draw_clusters()
            self.draw_flags()
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 128, 0, 64))
            self.screen.blit(overlay, (0, 0))
            main_text = "Game Won, Press 'R' to Restart"
        else:
            self.draw_clusters()
            self.draw_cells_bayes()
            self.draw_markers()
            self.draw_lines()
            blur_bg(self.screen, sigma=0.32)
            return  # Skip overlay and text rendering in regular gameplay
        self.draw_lines()

        # Overlay and main text rendering for game-over or game-won states
        blur_bg(self.screen, sigma=2)
        text_lines = [
            main_text,
            f"Current Level: {self.level.capitalize()}",
            "Press 1 - Easy",
            "Press 2 - Intermediate",
            "Press 3 - Hard",
            "Press 4 - Extreme",
            "Press H in game to Toggle Help",
        ]

        # Dynamically position text lines
        start_y = self.height // 4
        end_y = (3 * self.height) // 4
        vertical_spacing = (end_y - start_y) // (len(text_lines) - 1)
        for i, text in enumerate(text_lines):
            text_surface = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.width // 2, start_y + i * vertical_spacing))
            self.screen.blit(text_surface, text_rect)

    def draw_clusters(self):
        for cluster in find_clusters(self.board.minefield["state"], COVERED):
            draw_polygon_with_holes(
                self.screen,
                rects_to_polygon(get_rects_from_cluster(cluster, CELL_SIZE, BORDER_SIZE, LINE_WIDTH)),
                cell_color,
                background_color,
                CELL_SIZE,
            )
        blur_bg(self.screen, sigma=0.8)

    def draw_mines(self):
        for row, col in self.board.mines:
            self.screen.blit(
                self.scaled_mine_image,
                (
                    BORDER_SIZE
                    + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH)
                    + 1
                    + self.scaled_mine_image.get_width() // 2,
                    BORDER_SIZE
                    + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH)
                    + 1
                    + self.scaled_mine_image.get_height() // 2,
                ),
            )

    def draw_flags(self):
        for row, col in self.board.mines:
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

    def draw_markers(self):

        for row, col in self.flagged:
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
