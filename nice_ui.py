import os
import sys
import random
import asyncio
import numpy as np
import pygame as pg
import pygame.gfxdraw
import matplotlib.colors as mcolors
from game_engine import Minesweeper
from shapely.ops import unary_union
from shapely.geometry import Polygon
from typing import Any, Dict, List, Tuple, Callable

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

levels = {
    0: "test",
    1: "easy",
    2: "intermediate",
    3: "hard",
}


def get_custom_rgb(value: float) -> Tuple[int, int, int]:
    """
    Custom Green-Yellow-Orange-Red colormap.
    Parameters:
        value (float): A value between 0 and 1.
    Returns:
        Tuple[int, int, int]: A tuple of (R, G, B) values scaled to 0-255.
    """
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1")
    # Define the custom colormap: Green -> Yellow -> Orange -> Red
    colors = [
        (0, 1, 0),  # Green
        (1, 1, 0),  # Yellow
        (1, 0.5, 0),  # Orange
        (1, 0, 0),  # Red
    ]
    custom_colormap = mcolors.LinearSegmentedColormap.from_list("GreenYellowOrangeRed", colors)
    # Get RGB from the custom colormap
    rgb = np.array(custom_colormap(value)[:3]) * 255
    # return tuple(rgb.astype(int))
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
        queue: List[Tuple[int, int]] = [(start_r, start_c)]
        visited[start_r, start_c] = True
        while queue:
            r, c = queue.pop(0)
            cluster.append((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and board[nr, nc] == flag:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
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
        self.init_game()

    def init_game(self):
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
        self.font: pg.font.Font = pg.font.SysFont("orbitronmedium", font_size)
        self.clock: pg.time.Clock = pg.time.Clock()
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

        if key == pg.K_0:
            self.level = levels[0]
            self.reset_game()
        if key == pg.K_1:
            self.level = levels[1]
            self.reset_game()

        if key == pg.K_2:
            self.level = levels[2]
            self.reset_game()

        if key == pg.K_3:
            self.level = levels[3]
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
        self.draw_clusters()
        self.draw_mines()
        self.draw_cells_bayes()
        self.draw_lines()
        # Check for game over or won condition
        if self.board.game_won or self.board.game_over:
            # Create a transparent overlay surface
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent black background
            self.screen.blit(overlay, (0, 0))

            # Render text
            font = self.font  # Use default font, size 60
            if self.board.game_over:
                text = "Game Over, Press 'R' to Restart"
            else:
                text = "Game Won, Press 'R' to Restart"
            text_surface = font.render(text, True, (255, 255, 255))  # White text
            text_rect = text_surface.get_rect(center=self.screen.get_rect().center)

            # Draw the text
            self.screen.blit(text_surface, text_rect)

    def draw_clusters(self):
        # Draw clusters
        board = self.board.minefield["state"]
        clusters = find_clusters(board, COVERED)
        for cluster in clusters:
            rects = get_rects_from_cluster(cluster)
            polygon = rects_to_polygon(rects)
            draw_polygon_with_holes(self.screen, polygon, cell_color, background_color)

    def draw_mine(self, x: int, y: int, size: int):
        rect = pygame.Rect(x, y, size, size)
        polygon = Polygon([rect.topleft, rect.topright, rect.bottomright, rect.bottomleft])
        resolution = max(16, int(polygon.length / 10))
        dilate_distance = CELL_SIZE // 9

        # Smooth the exterior of the polygon
        smoothed_exterior = (
            polygon.buffer(dilate_distance, cap_style=1, join_style=1, resolution=resolution)
            .buffer(-dilate_distance * 3.0, cap_style=1, join_style=1, resolution=resolution)
            .buffer(dilate_distance, cap_style=1, join_style=1, resolution=resolution)
        )

        # Draw the smoothed exterior
        pg.gfxdraw.aapolygon(
            self.screen,
            list(map(tuple, smoothed_exterior.exterior.coords)),
            pygame.Color("grey55"),
        )
        pg.gfxdraw.filled_polygon(
            self.screen,
            list(map(tuple, smoothed_exterior.exterior.coords)),
            pygame.Color("grey55"),
        )
        # pygame.draw.rect(self.screen, pygame.Color("red"), rect)

        # Draw the black center circle
        center_x, center_y = x + size // 2, y + size // 2
        center_radius = size // 3
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, center_radius, pygame.Color("black"))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, center_radius, pygame.Color("black"))

        # Line properties
        line_length = size // 2.4
        line_thickness = size // 12
        tip_radius = size // 30

        # Draw the '+' lines
        # Horizontal line
        pygame.draw.line(
            self.screen,
            pygame.Color("black"),
            (center_x - int(line_length), center_y),
            (center_x + int(line_length), center_y),
            line_thickness,
        )
        # Vertical line
        pygame.draw.line(
            self.screen,
            pygame.Color("black"),
            (center_x, center_y - int(line_length)),
            (center_x, center_y + int(line_length)),
            line_thickness,
        )

        # Rotate the '+' lines by 45 degrees to form an 'X'
        diagonal_offset = int(0.95 * line_length * np.sqrt(2) / 2)  # Diagonal length for 45-degree rotation

        # Diagonal line (top-left to bottom-right)
        pygame.draw.line(
            self.screen,
            pygame.Color("black"),
            (center_x - diagonal_offset, center_y - diagonal_offset),
            (center_x + diagonal_offset, center_y + diagonal_offset),
            tip_radius * 4,
        )
        # Diagonal line (bottom-left to top-right)
        pygame.draw.line(
            self.screen,
            pygame.Color("black"),
            (center_x - diagonal_offset, center_y + diagonal_offset),
            (center_x + diagonal_offset, center_y - diagonal_offset),
            tip_radius * 4,
        )

        # Draw small filled circles at the ends of the diagonal lines
        tips = [
            (center_x - diagonal_offset, center_y - diagonal_offset),  # Top-left tip
            (
                center_x + diagonal_offset,
                center_y + diagonal_offset,
            ),  # Bottom-right tip
            (center_x - diagonal_offset, center_y + diagonal_offset),  # Bottom-left tip
            (center_x + diagonal_offset, center_y - diagonal_offset),  # Top-right tip
        ]
        for tip in tips:
            pygame.gfxdraw.aacircle(self.screen, tip[0], tip[1], tip_radius, pygame.Color("black"))
            pygame.gfxdraw.filled_circle(self.screen, tip[0], tip[1], tip_radius, pygame.Color("black"))

    def draw_mines(self):
        if self.board.game_over:
            for row, col in self.board.mines:
                x = BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                y = BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                self.draw_mine(x, y, CELL_SIZE)

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

                # Draw covered cells with probabilities (if available)
                if cell["state"] == self.board.states.COVERED.value:
                    if self.probability is not None:
                        probability = self.probability[row, col]
                        text_surface = self.font.render(f"{probability:.1f}", True, get_custom_rgb(probability))
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
