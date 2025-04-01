from __future__ import annotations
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, Colormap
from typing import Tuple

import plot_functions

# ===================================================
# Class - Color scheme
# ===================================================


class Color_Scheme:
    """
    Class with colors for network simulation.
    """
    def __init__(self, show=False) -> None:
        self.colors_lst, self.red, self.cmap = self.color_scheme(show)

    def color_scheme(self, show: bool = False) -> Tuple[list[str], str, Colormap]:
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
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [colors_lst[1], colors_lst[2], colors_lst[0]],
                                                        N=256)

        if show:
            plot_functions.plot_colors(custom_cmap, red)
        return colors_lst, red, custom_cmap
