from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Colormap

from typing import Tuple, List, Union, Optional
from numpy.typing import NDArray


# ==================================
# color scheme
# ==================================


def color_scheme(show: bool = False) -> Tuple[list[str], str, Colormap]:
    """
    define color scheme and return main colors, main red color and a colormap

    inputs:
    show - boolean of whether to plot colormap and red color

    outputs:
    colors      - list of strings names of colors in hexadecimal (#RRGGBB)
    red         - str, hexadecimal for soft red
    custom_cmap - matplotlibcolors.Colormap of 256 colors on a scale using "colors"
    """

    # Define the custom color scheme as a colormap

    colors_lst = ['#4500E0', '#54CCE0', '#CD23E1', '#9EE1B1', '#E04F68']
    red = '#E04F68'

    # Create the custom colormap for the gradient
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [colors_lst[1], colors_lst[0], colors_lst[2]], N=256)

    if show:
        # Create a gradient and plot it with log scale on the y-axis
        plt.figure(figsize=(8, 4))

        # Generate a vertical gradient and plot with log scale
        gradient = np.linspace(0, 1, 256).reshape(256, 1)  # Vertical gradient

        # Plot the custom gradient
        plt.subplot(1, 2, 1)
        plt.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 1, 256])
        # plt.imshow(gradient, cmap=custom_cmap)
        plt.title("Custom Color Gradient")
        plt.xticks([])  # Remove x ticks
        plt.yticks([])  # Remove y ticks

        # Plot the solid red block using a 1x1 matrix with the red color mapped
        plt.subplot(1, 2, 2)
        plt.imshow([[1]], aspect='auto', cmap=LinearSegmentedColormap.from_list('red_cmap', [red, red]),
                   extent=[0, 1, 1, 256])
        plt.title("Solid Red Color")
        plt.xticks([])  # Remove x ticks
        plt.yticks([])  # Remove y ticks

        plt.tight_layout()
        plt.show()
    return colors_lst, red, custom_cmap
