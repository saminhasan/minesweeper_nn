from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Callable, Set
from shapely.ops import unary_union
from shapely.geometry import Polygon
import numpy as np
import pygame as pg
import pygame.gfxdraw
from collections import deque


def blur_bg(screen, sigma=0.5):
    """Apply a Gaussian filter to each color channel."""
    # Fetch the color planes once as a 3D array
    pixels = pygame.surfarray.pixels3d(screen)  # Shape: (width, height, 3)

    # Apply the Gaussian filter separately to each channel
    for channel in range(3):  # 0: Red, 1: Green, 2: Blue
        gaussian_filter(pixels[:, :, channel], sigma=sigma, mode="nearest", output=pixels[:, :, channel])


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


def get_rects_from_cluster(cluster: List[Tuple[int, int]], CELL_SIZE, BORDER_SIZE, LINE_WIDTH) -> List[pg.Rect]:
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
    surface: pg.Surface, polygon: Polygon, fill_color: Tuple[int, int, int], background_color: Tuple[int, int, int], CELL_SIZE
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
