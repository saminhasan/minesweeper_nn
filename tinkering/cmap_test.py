from typing import Tuple


class CustomColormap:
    def __init__(self):
        pass

    def get_rgb(self, value: float) -> Tuple[int, int, int]:
        """
        Maps a value in [0, 1] to an RGB color.
        Parameters:
            value (float): A value between 0 and 1.
        Returns:
            Tuple[int, int, int]: RGB color as integers in the range [0, 255].
        """
        if not 0 <= value <= 1:
            raise ValueError("Value must be between 0 and 1")

        return (int(255 * (2 * value)) if value <= 0.5 else 255, 255 if value <= 0.5 else int(255 * (2 * (1 - value))), 0)


import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Generate some example data
    data = np.linspace(0, 1, 10000).reshape(100, 100)  # Normalized data

    # Instantiate the custom colormap
    colormap = CustomColormap()

    # Map the data to RGB colors using the custom colormap
    rgb_data = np.array([colormap.get_rgb(value) for value in data.ravel()])
    rgb_data = rgb_data.reshape(*data.shape, 3) / 255  # Reshape and normalize to [0, 1] for matplotlib

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the data with the custom colormap
    im = ax.imshow(rgb_data, interpolation="nearest")

    # Add a title and axis labels
    ax.set_title("Custom Green-Yellow-Orange-Red Colormap")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Add a colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("jet"))  # Placeholder for colorbar
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Normalized Value (0 to 1)")

    # Show the plot
    plt.show()
