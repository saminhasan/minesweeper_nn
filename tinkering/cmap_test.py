from typing import Tuple


class CustomColormap:
    def __init__(self):
        """
        Initializes a Green-Yellow-Orange-Red colormap with pre-defined RGB values.
        """
        self.colors = [
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (255, 128, 0),  # Orange
            (255, 0, 0),  # Red
        ]
        self.num_segments = len(self.colors) - 1

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

        # Scale value to the number of segments
        scaled_value = value * self.num_segments
        idx = int(scaled_value)  # Find the segment index
        t = scaled_value - idx  # Fractional part within the segment

        # Get the start and end colors for interpolation
        color1 = self.colors[idx]
        color2 = self.colors[min(idx + 1, self.num_segments)]

        # Linear interpolation for each RGB channel
        r = int((1 - t) * color1[0] + t * color2[0])
        g = int((1 - t) * color1[1] + t * color2[1])
        b = int((1 - t) * color1[2] + t * color2[2])

        return r, g, b


import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Generate some example data
    data = np.linspace(0, 1, 10000).reshape(100, 100)  # Normalized data

    # Instantiate the custom colormap
    colormap = CustomColormap()

    # Map the data to RGB colors using the custom colormap
    rgb_data = np.array([colormap.get_rgb(value) for value in data.ravel()])
    rgb_data = (
        rgb_data.reshape(*data.shape, 3) / 255
    )  # Reshape and normalize to [0, 1] for matplotlib

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
