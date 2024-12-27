import pygame
import pygame.gfxdraw
import sys
import math
import numpy as np

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mine Drawing")


def draw_background():
    """Draws a black background on the screen."""
    screen.fill(BLACK)


import pygame
import pygame.gfxdraw
import math


def draw_mine(x: int, y: int, size: int):
    # Draw the red bounding rectangle for reference (if needed)
    rect = pygame.Rect(x, y, size, size)
    pygame.draw.rect(screen, RED, rect)

    # Draw the black center circle
    center_x, center_y = x + size // 2, y + size // 2
    center_radius = size // 3
    pygame.gfxdraw.aacircle(screen, center_x, center_y, center_radius, BLACK)
    pygame.gfxdraw.filled_circle(screen, center_x, center_y, center_radius, BLACK)

    # Line properties
    line_length = size // 2.4
    line_thickness = size // 12
    tip_radius = size // 30

    # Draw the '+' lines
    # Horizontal line
    pygame.draw.line(
        screen,
        BLACK,
        (center_x - int(line_length), center_y),
        (center_x + int(line_length), center_y),
        line_thickness,
    )
    # Vertical line
    pygame.draw.line(
        screen,
        BLACK,
        (center_x, center_y - int(line_length)),
        (center_x, center_y + int(line_length)),
        line_thickness,
    )

    # Rotate the '+' lines by 45 degrees to form an 'X'
    diagonal_offset = int(
        0.95 * line_length * math.sqrt(2) / 2
    )  # Diagonal length for 45-degree rotation

    # Diagonal line (top-left to bottom-right)
    pygame.draw.line(
        screen,
        BLACK,
        (center_x - diagonal_offset, center_y - diagonal_offset),
        (center_x + diagonal_offset, center_y + diagonal_offset),
        2,
    )
    # Diagonal line (bottom-left to top-right)
    pygame.draw.line(
        screen,
        BLACK,
        (center_x - diagonal_offset, center_y + diagonal_offset),
        (center_x + diagonal_offset, center_y - diagonal_offset),
        2,
    )

    # Draw small filled circles at the ends of the diagonal lines
    tips = [
        (center_x - diagonal_offset, center_y - diagonal_offset),  # Top-left tip
        (center_x + diagonal_offset, center_y + diagonal_offset),  # Bottom-right tip
        (center_x - diagonal_offset, center_y + diagonal_offset),  # Bottom-left tip
        (center_x + diagonal_offset, center_y - diagonal_offset),  # Top-right tip
    ]
    for tip in tips:
        pygame.gfxdraw.aacircle(screen, tip[0], tip[1], tip_radius, BLACK)
        pygame.gfxdraw.filled_circle(screen, tip[0], tip[1], tip_radius, BLACK)


def main():
    """Main game loop."""
    clock = pygame.time.Clock()

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the background
        draw_background()

        # Example usage: Draw some mines
        draw_mine(200, 300, 50)
        draw_mine(400, 300, 70)
        draw_mine(600, 300, 100)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)


if __name__ == "__main__":
    main()
