import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import ctypes
import asyncio
import numpy as np
import pygame as pg
import pygame.gfxdraw
from collections import deque
from game_engine import Minesweeper
from shapely.ops import unary_union
from shapely.geometry import Polygon
from typing import Dict, List, Tuple, Callable

# Constants with explicit type annotations
CELL_SIZE: int = 48
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


def get_custom_rgb(value: float) -> Tuple[int, int, int]:
    """
    Generate an RGB color from a custom Green-to-Yellow-to-Red colormap based on a value.

    The colormap transitions smoothly in the HSV color space, varying only the Hue (H) value,
    while keeping Saturation (S) and Value (V) constant. The resulting colors are converted
    to RGB for practical use. This colormap is designed to visually represent safety (green)
    and danger (red), with yellow as a midpoint.

    Parameters:
        value (float): A value between 0 and 1, where:
                       - 0 maps to green (safe),
                       - 0.5 maps to yellow,
                       - 1 maps to red (danger).

    Returns:
        Tuple[int, int, int]: A tuple of (R, G, B) values scaled to 0-255.

    Raises:
        ValueError: If the input value is not between 0 and 1.

    Example:
        >>> get_custom_rgb(0.0)
        (0, 255, 0)  # Green
        >>> get_custom_rgb(0.5)
        (255, 255, 0)  # Yellow
        >>> get_custom_rgb(1.0)
        (255, 0, 0)  # Red
    """
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1")
    return (int(255 * (2 * value)) if value <= 0.5 else 255, 255 if value <= 0.5 else int(255 * (2 * (1 - value))), 0)


def find_clusters(board: np.ndarray, flag: int) -> List[List[Tuple[int, int]]]:
    """
    Identify clusters of covered cells on the board.

    Parameters:
        board (np.ndarray): A 2D NumPy array representing the board state.
        flag (int): The value to identify clusters of cells.

    Returns:
        List[List[Tuple[int, int]]]: A list of clusters, where each cluster is a list of cell coordinates (row, col).
    """
    rows, cols = board.shape
    visited = np.zeros_like(board, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Define the 4 possible directions

    def flood_fill_iterative(start_r: int, start_c: int) -> List[Tuple[int, int]]:
        cluster: List[Tuple[int, int]] = []
        queue: deque = deque([(start_r, start_c)])  # Use deque for the queue
        visited[start_r, start_c] = True

        while queue:
            r, c = queue.popleft()  # Deque's popleft is O(1)
            cluster.append((r, c))

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and board[nr, nc] == flag:
                    visited[nr, nc] = True
                    queue.append((nr, nc))  # Deque's append is O(1)

        return cluster

    # List comprehension for the clusters
    return [
        flood_fill_iterative(row, col)
        for row in range(rows)
        for col in range(cols)
        if board[row, col] == flag and not visited[row, col]
    ]


def get_rects_from_cluster(cluster: List[Tuple[int, int]]) -> List[pg.Rect]:
    """
    Get pygame Rect objects for a given cluster of cell coordinates.

    Parameters:
        cluster (List[Tuple[int, int]]): A list of tuples representing the (row, column) coordinates of the cluster.

    Returns:
        List[pg.Rect]: A list of pygame Rect objects for the cluster.
    """
    return [
        pg.Rect(
            BORDER_SIZE + c * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1,
            BORDER_SIZE + r * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1,
            CELL_SIZE,
            CELL_SIZE,
        )
        for r, c in cluster
    ]


def rects_to_polygon(rects: List[pg.Rect]) -> Polygon:
    """
    Convert a list of pygame Rect objects into a single Shapely Polygon.

    Parameters:
        rects (List[pg.Rect]): A list of pygame Rect objects.

    Returns:
        Polygon: A Shapely Polygon representing the union of all rectangles.
    """
    return unary_union(
        [
            Polygon(
                [
                    rect.topleft,
                    rect.topright,
                    rect.bottomright,
                    rect.bottomleft,
                ]
            )
            for rect in rects
        ]
    )


def draw_polygon_with_holes(
    surface: pg.Surface,
    polygon: Polygon,
    fill_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int],
) -> None:
    """
    Draw a Shapely Polygon with holes on a pygame surface.

    Parameters:
        surface (pg.Surface): The pygame surface to draw on.
        polygon (Polygon): A Shapely Polygon object, potentially with holes.
        fill_color (Tuple[int, int, int]): RGB color to fill the exterior of the polygon.
        background_color (Tuple[int, int, int]): RGB color to fill the holes (interiors) of the polygon.
    """
    # Compute smoothing parameters
    resolution = max(16, int(polygon.length / 10))
    dilate_distance = CELL_SIZE // 9

    # Smooth the exterior of the polygon
    smoothed_exterior = (
        polygon.buffer(dilate_distance, cap_style=1, join_style=1, resolution=resolution)
        .buffer(-dilate_distance * 3.0, cap_style=1, join_style=1, resolution=resolution)
        .buffer(dilate_distance, cap_style=1, join_style=1, resolution=resolution)
    )

    # Draw the smoothed exterior
    pg.gfxdraw.aapolygon(surface, list(map(tuple, smoothed_exterior.exterior.coords)), fill_color)
    pg.gfxdraw.filled_polygon(surface, list(map(tuple, smoothed_exterior.exterior.coords)), fill_color)

    # Process and draw each hole (interior)
    for interior in polygon.interiors:
        smoothed_hole = (
            Polygon(interior.coords)
            .buffer(dilate_distance * 1.5, cap_style=1, join_style=1, resolution=resolution)
            .buffer(-dilate_distance * 2, cap_style=1, join_style=1, resolution=resolution)
            .buffer(dilate_distance * 1.5, cap_style=1, join_style=1, resolution=resolution)
        )
        pg.gfxdraw.aapolygon(surface, list(map(tuple, smoothed_hole.exterior.coords)), background_color)
        pg.gfxdraw.filled_polygon(surface, list(map(tuple, smoothed_hole.exterior.coords)), background_color)


class GUI:
    def __init__(self, level: str):
        self.level = levels[level]  # Store the level for resetting the game
        self.init = True
        self.init_game()

    def init_game(self):
        if self.init:
            pg.quit()

        """
        Initialize the Minesweeper GUI.

        Parameters:
            level (str): The difficulty level of the Minesweeper game (e.g., 'easy', 'medium', 'hard').
        """
        self.board = Minesweeper(self.level)  # Create a Minesweeper board
        # Solve the minefield and get the solution and probabilities
        _, self.probability = self.board.solve_minefield()

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
        self.mine_image: pg.Surface = pg.image.load("minesweeper_icon.png")  # Load icon image
        # Scale the mine image to fit the cell size
        self.scaled_mine_image = pg.transform.scale(self.mine_image, (CELL_SIZE, CELL_SIZE))
        pg.display.set_icon(self.mine_image)  # Set the icon for the window

        self.font: pg.font.Font = pg.font.SysFont("orbitronmedium", font_size)
        self.clock: pg.time.Clock = pg.time.Clock()

        # Calculate the screen's center position for the window
        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{(screen_width - self.width) // 2},{(screen_height - self.height) // 2}"
        self.screen: pg.Surface = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Minesweeper")

        # Key event handlers
        self.key_event_handlers: Dict[int, Callable[[], None]] = {
            pg.K_ESCAPE: self.quit,
        }

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
        if key == pg.K_r:  # Check if 'R' key is pressed
            self.reset_game()

        if key in [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4]:
            self.level = levels[key - pg.K_0]
            self.reset_game()
        else:
            handler = self.key_event_handlers.get(key)
            if handler:
                handler()

    def handle_mouse_event(self, event: pg.event.Event) -> None:
        """
        Handle mouse events for the game.

        Parameters:
            event (pg.event.Event): The pygame mouse event object.
        """
        if not (self.board.game_over or self.board.game_won):
            if event.button == pg.BUTTON_LEFT:  # Left click
                mouse_x, mouse_y = event.pos  # Get mouse position
                # Pixel space to cell space
                col = (mouse_x - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
                row = (mouse_y - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
                if 0 <= row < self.board.n_rows and 0 <= col < self.board.n_cols:
                    if self.board.minefield[row, col]["mine_count"] == -1:
                        self.board.reveal_all_mines()
                        print("Game Over")
                    elif not self.board.game_over:
                        self.board.reveal(row, col)
                        _, self.probability = self.board.solve_minefield()

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
            overlay.fill((128, 0, 0, 128))  # Semi-transparent black background
            self.screen.blit(overlay, (0, 0))

            # Display game over message
            main_text = "Game Over, Press 'R' to Restart"
        elif self.board.game_won:
            # Handle game won
            self.draw_clusters()
            self.draw_cells_bayes()

            # Create a transparent overlay surface
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 128, 0, 128))  # Semi-transparent green background
            self.screen.blit(overlay, (0, 0))

            # Display game won message
            main_text = "Game Won, Press 'R' to Restart"
        else:
            # Handle normal game state
            self.draw_clusters()
            self.draw_cells_bayes()
            self.draw_lines()
            return  # Skip the rest since no overlay or text is needed

        # Render main text
        main_text_surface = self.font.render(main_text, True, (255, 255, 255))  # White text
        main_text_rect = main_text_surface.get_rect(center=(self.width // 2, self.height // 3))

        # Render level text
        level_text = f"Current Level: {self.level.capitalize()}"  # Capitalize level string
        level_surface = self.font.render(level_text, True, (255, 255, 255))
        level_rect = level_surface.get_rect(center=(self.width // 2, self.height // 2))

        # Render options text (split into lines)
        options_lines = ["Press 1 - Easy", "Press 2 - Intermediate", "Press 3 - Hard", "Press 4 - Extreme"]
        options_surfaces = [self.font.render(line, True, (255, 255, 255)) for line in options_lines]
        options_rects = [
            surface.get_rect(center=(self.width // 2, (2 * self.height) // 3 + i * 30))  # Adjust vertical spacing
            for i, surface in enumerate(options_surfaces)
        ]

        # Draw everything
        self.screen.blit(main_text_surface, main_text_rect)
        self.screen.blit(level_surface, level_rect)
        [self.screen.blit(surface, rect) for surface, rect in zip(options_surfaces, options_rects)]

        self.draw_lines()

    def draw_clusters(self):
        [
            draw_polygon_with_holes(
                self.screen, rects_to_polygon(get_rects_from_cluster(cluster)), cell_color, background_color
            )
            for cluster in find_clusters(self.board.minefield["state"], COVERED)
        ]

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

    def draw_cells_bayes(self):
        # Draw cells with numbers or probabilities
        for row in range(self.board.n_rows):
            for col in range(self.board.n_cols):
                x = BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                y = BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                cell = self.board.minefield[row, col]

                # Draw uncovered cells with mine counts
                if cell["state"] == self.board.states.UNCOVERED.value:
                    if cell["mine_count"] > 0:
                        text_surface = self.font.render(f"{cell['mine_count']}", True, text_color)
                        text_rect = text_surface.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                        self.screen.blit(text_surface, text_rect)

                if cell["state"] == self.board.states.COVERED.value:
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


async def main():
    game = GUI(1)

    try:
        while game.running:
            game.handle_events()
            game.draw()
            pg.display.update()
            await asyncio.sleep(0)  # Yield control to the event loop
    except Exception as e:
        print(e)
    finally:
        pg.quit()


if __name__ == "__main__":
    asyncio.run(main())
