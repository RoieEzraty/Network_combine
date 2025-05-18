from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Dict, Any, Union, Optional
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from Network_State import Network_State
    from Big_Class import Big_Class

import colors, statistics

# ================================
# functions for plots
# ================================


def plot_importants(BigClass: "Big_Class", movmean_loss: bool = False, include_network: Optional[bool] = False,
                    node_labels: bool = False) -> None:
    """
    one plot with 4 subfigures of
    1) mean absolute value of loss in time
    2) inputs and outputs in the update modality, in time
    3) resistances in time
    4) Network structure, from networkx pos_lattice

    inputs:
    BigClass        - Class instance containing User_Variables, Network_Structure, etc.
    movmean_loss    - boolean of whether to smoothen loss with moving mean
    include_network - boolean of whether to plot network
    node_label      - boolean of whether to plot node number

    outputs:
    1 matplotlib plot
    """

    Nin = BigClass.Variabs.Nin
    Nout = BigClass.Variabs.Nout
    t = BigClass.State.t

    # Set the custom color cycle globally without cycler
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    legend2 = []
    if Nout == 1:
        legend2.append(r'$y\,\mathrm{update}$')
    else:
        legend2 += [rf'$y_{{{j+1}}}\,\mathrm{{update}}$' for j in range(Nout)]

    # Input updates: x, x_1, x_2, ...
    if Nin == 1:
        legend2.append(r'$x\,\mathrm{update}$')
    else:
        legend2 += [rf'$x_{{{i+1}}}\,\mathrm{{update}}$' for i in range(Nin)]

    if include_network:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17, 3))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.75, 3))

    # Loss
    for t in range(t):
        if t % len(BigClass.Variabs.dataset) == 0 and t != 0 and BigClass.Variabs.task_type != 'Regression':
            ax1.axvline(x=t, color='red', linestyle='--', linewidth=1)
    if movmean_loss:
        movmean_loss_t = statistics.mov_ave(BigClass.State.loss_scalar_in_t, 16)
        ax1.plot(np.abs(movmean_loss_t[1:]))
    else:
        ax1.plot(BigClass.State.loss_scalar_in_t[1:])
    # ax1.plot(np.mean(np.mean(np.abs(BigClass.State.loss_in_t[1:]), axis=1), axis=1))
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1)
    ax1.set_title(r'$\|\mathcal{L}\|$')
    ax1.set_xlabel('t')

    # Update modality
    ax2.plot(BigClass.State.output_update_in_t[1:])
    ax2.plot(BigClass.State.input_update_in_t[1:])
    if BigClass.Variabs.access_interNodes:
        ax2.plot(BigClass.State.inter_update_in_t[1:])
    ax2.set_title('"Update" modality pressure')
    ax2.set_xlabel('t')
    if legend2 and len(legend2) < 6:
        ax2.legend(legend2)

    # Resistances
    ax3.plot(BigClass.State.R_in_t[1:])
    ax3.set_title(r'$R$')
    ax3.set_xlabel('t')

    # Network structure
    if include_network:
        if BigClass.NET.NET is not None:
            plotNetStructure(NET=BigClass.NET.NET,
                             BigClass=BigClass,
                             pos_lattice=BigClass.NET.pos_lattice,
                             node_labels=node_labels,
                             R_reordered=BigClass.NET.R_reordered,
                             u_reordered=BigClass.NET.u_reordered,
                             p_reordered=BigClass.NET.p_reordered,
                             ax=ax4  # Pass the subplot axis
                             )
        else:
            print('no NET assigned in input')
    plt.show()


def plotNetStructure(NET: nx.DiGraph, BigClass: "Big_Class",
                     pos_lattice: Dict[Any, Tuple[float, float]], node_labels: bool = False,
                     R_reordered: NDArray[np.float_] = np.array([]),
                     u_reordered: NDArray[np.float_] = np.array([]),
                     p_reordered: NDArray[np.float_] = np.array([]), ax: Optional[plt.Axes] = None) -> None:
    """
    Plots the structure (nodes and edges) of networkx NET
    Arrows represent flow direction, arrow width = flow magnitude, color = R (low-cyan, high-purple)

    input:
    NET         - networkx net of nodes and edges
    BigClass    - Class instance containing User_Variables, Network_Structure, etc.
    pos_lattice - dict of positions of nodes from NET.nodes
    node_labels - boolean, show node number in plot or not
    R_reordered - array of values used to determine edge colors
    u_reordered - array of values used to determine edge widths and flow direction
    p_reordered - array of values used to determine node colors

    output:
    matplotlib plot of network structure
    """
    colors_lst = BigClass.Colorscheme.colors_lst
    # Determine edge colors from resistances
    if R_reordered.size > 0:
        R_reordered_normalized = 4 * R_reordered / np.max(R_reordered)  # resistances
        edge_colors = [BigClass.Colorscheme.cmap(value) for value in R_reordered_normalized]
    else:
        edge_colors = [colors_lst[0] for _ in range(len(NET.edges))]

    # Determine edge widths and directions from flow u
    if u_reordered.size > 0:
        edge_widths = 3 * np.abs(u_reordered)/np.max(np.abs(u_reordered))  # Widths based on absolute flow
        edge_directions = [1 if flow > 0 else -1 for flow in u_reordered]  # Positive or negative flow
    else:
        edge_widths = [1.0 for _ in range(len(NET.edges))]
        edge_directions = [1 for _ in range(len(NET.edges))]  # Default all positive

    # Determine node colors from pressure p
    if p_reordered.size > 0:
        p_reordered_normalized = p_reordered / np.max(p_reordered)
        node_colors = [BigClass.Colorscheme.cmap(value) for value in p_reordered_normalized]
    else:
        node_colors = [colors_lst[0] for _ in range(len(NET.nodes))]

    # Create or use the specified axis
    ax = ax or plt.gca()

    # Draw edges with arrows
    for (u, v), color, width, direction in zip(NET.edges, edge_colors, edge_widths, edge_directions):
        if direction > 0:  # Positive flow
            nx.draw_networkx_edges(NET, pos_lattice, edgelist=[(u, v)], edge_color=[color], width=width,
                                   connectionstyle="arc3,rad=0.0", arrowstyle="-|>", arrows=True, ax=ax)
        else:  # Negative flow (reverse direction)
            nx.draw_networkx_edges(NET, pos_lattice, edgelist=[(u, v)], edge_color=[color], width=width,
                                   connectionstyle="arc3,rad=0.0", arrowstyle="<|-", arrows=True, ax=ax)

    # Draw nodes
    # nx.draw_networkx_nodes(NET, pos=pos_lattice, node_color=node_colors, node_size=100)
    nx.draw_networkx_nodes(NET, pos=pos_lattice, node_color=node_colors, node_size=20)

    # Highlight input nodes
    nx.draw_networkx_nodes(NET,
                           pos=pos_lattice,
                           nodelist=BigClass.Strctr.input_nodes_arr,
                           node_color="none",  # Hollow circle
                           edgecolors="k",  # black border
                           node_size=200,  # Adjust size as needed
                           linewidths=2)  # Thickness of the border

    # Highlight output nodes
    nx.draw_networkx_nodes(NET,
                           pos=pos_lattice,
                           nodelist=BigClass.Strctr.output_nodes_arr,
                           node_color="none",  # Hollow circle
                           edgecolors="grey",  # Grey border
                           node_size=200,  # Adjust size as needed
                           linewidths=2)  # Thickness of the border

    # Draw labels (if enabled)
    if node_labels:
        nx.draw_networkx_labels(NET, pos_lattice)

    # Show the plot
    plt.show()
    print('NET is ready')


def plot_colors(custom_cmap, red):
    """
    Plot a visualization of a custom colormap and a solid red color block.

    This function displays two side-by-side color panels:
    - A vertical gradient using the provided `custom_cmap`.
    - A solid red color using a single RGB value.

    It is useful for visually verifying the appearance of colormaps and individual color choices
    used for plotting or presentation design.

    Parameters
    ----------
    custom_cmap : matplotlib.colors.Colormap
        A custom colormap to be visualized as a vertical gradient.
    red : tuple or list of float
        An RGB triplet (each value in [0, 1]) representing a solid red color to be displayed.

    output:
    matplotlib plot
    """
    # Create a gradient and plot it with log scale on the y-axis
    plt.figure(figsize=(8, 4))

    # Generate a vertical gradient and plot with log scale
    gradient = np.linspace(0, 1, 256).reshape(256, 1)  # Vertical gradient

    # Plot the custom gradient
    plt.subplot(1, 2, 1)
    plt.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, 1, 256])
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


def plot_accuracy(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                  dataset_len: np.int_) -> None:
    """
    Plots the accuracy in time for the Iris classification task

    input:
    t_final        - int, final time step
    t_for_accuracy - array of ints, times during simulation when accuracy was calculated
    accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
    dataset_len    - length of dataset used, for Iris it is 150

    output:
    plot of accuracy a.f.o time
    """
    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1)

    # plot accuracy a.f.o time
    plt.plot(t_for_accuracy[1:], accuracy_in_t[1:], label='accuracy')  # plot from t=1 since starts at 0.3 and not 0

    # axes
    plt.xlabel('t', fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Accuracy', fontsize=14)  # Set y-axis label with font size
    plt.title('Accuracy Over Time', fontsize=16)  # Set title with font size
    plt.ylim([0, 1])


def plot_Power(State: "Network_State") -> None:
    """
    Plot the normalized power consumption over time on a logarithmic scale.

    This function visualizes the temporal evolution of normalized power in the system,
    based on the simulation data stored in the `State` object.

    Parameters
    ----------
    State   - class instance of the state variables of network

    output:
    matplotlib plot
    """
    plt.plot(np.linspace(0, State.t, np.shape(State.Power_norm_in_t)[0]).T, State.Power_norm_in_t, color='blue')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{P}$')
    plt.yscale('log')


def plot_hist(vec: NDArray[np.float_], xLabel: str):
    """
    Plots the histogram of the vector "vec"

    input:
    vec    - 1D NDArray
    xLabel - str, naming the vec

    output:
    plot of histogram of vec
    """
    plt.figure(figsize=(4, 3))  # Set figure size
    colors_lst, red, custom_cmap = colors.color_scheme()

    plt.hist(vec, bins=16, color=colors_lst[0], alpha=0.7, edgecolor='black')  # Customize bins and style
    plt.xlabel(xLabel, fontsize=12)  # Label for x-axis
    plt.ylabel('Count', fontsize=12)  # Label for y-axis
    plt.xlim([0, 2.1])
    plt.ylim([0, np.size(vec)])
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()


# # NOT IN USE

# plot with measured pressures

def plot_importants_w_measured(BigClass: "Big_Class", M: Optional[NDArray[np.int_]] = None,
                               include_network: Optional[bool] = False, NET: Optional[nx.DiGraph] = None,
                               node_labels: bool = False) -> None:
    """
    one plot with 4 subfigures of
    1) output / desired - 1.
    2) inputs and outputs of the update modality
    3) resistances in time
    4) absolute mean value of loss in time

    inputs:
    State   - class instance of the state variables of network
    Variabs - class instance of the variables by the user
    desired - List of arrays of desired outputs given the drawn inputs and task matrix M
    M       - task matrix M under which desired output = M*input

    outputs:
    1 matplotlib plot
    """

    Nin = BigClass.Variabs.Nin
    Nout = BigClass.Variabs.Nout
    t = BigClass.State.t
    output_in_t = BigClass.State.output_in_t
    desired_in_t = BigClass.State.desired_in_t

    colors_lst, red, custom_cmap = colors.color_scheme()
    # Set the custom color cycle globally without cycler
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    if Nin == 1 and Nout == 1:  # 1by1, simplest
        legend1 = [r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$y\,\mathrm{update}$', r'$x\,\mathrm{update}$']
    elif Nin == 1 and Nout == 2:  # Allostery
        legend1 = [r'$\frac{y_1}{y_1\,\mathrm{desired}}$', r'$\frac{y_2}{y_2\,\mathrm{desired}}$']
        legend2 = [r'$y_1\,\mathrm{update}$', r'$y_2\,\mathrm{update}$', r'$x\,\mathrm{update}$']
    elif Nin == 2 and Nout == 1:  # Regression
        legend1 = [r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$y\,\mathrm{update}$', r'$x_1\,\mathrm{update}$', r'$x_2\,\mathrm{update}$']
    elif Nin == 2 and Nout == 3:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{update}$', r'$y\,\mathrm{update}$', r'$z\,\mathrm{update}$', r'$p_1\,\mathrm{update}$',
                   r'$p_2\,\mathrm{update}$']
    elif Nin == 2 and Nout == 2:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        if BigClass.Variabs.access_interNodes:
            legend2 = [r'$x\,\mathrm{update}$', r'$y\,\mathrm{update}$', r'$p_1\,\mathrm{update}$',
                       r'$p_2\,\mathrm{update}$', r'$\mathrm{inter1\,update}$', r'$\mathrm{inter2\,update}$']
        else:
            legend2 = [r'$x\,\mathrm{update}$', r'$y\,\mathrm{update}$', r'$p_1\,\mathrm{update}$',
                       r'$p_2\,\mathrm{update}$']
    elif Nin == 3 and Nout == 3:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{update}$', r'$y\,\mathrm{update}$', r'$z\,\mathrm{update}$', r'$p_1\,\mathrm{update}$',
                   r'$p_2\,\mathrm{update}$', r'$p_3\,\mathrm{update}$']
    elif BigClass.Variabs.task_type == 'Iris_classification':
        legend1 = [r'$\mathrm{Setosa}$', r'$\mathrm{Verisicolor}$', r'$\mathrm{Virginica}$']
        legend2 = [r'$\mathrm{Setosa\,update}$', r'$\mathrm{Verisicolor\,update}$',
                   r'$\mathrm{Virginica\,update}$', r'$p_1\,\mathrm{update}$', r'$p_2\,\mathrm{update}$',
                   r'$p_3\,\mathrm{update}$', r'$p_4\,\mathrm{update}$']
    else:
        legend1 = []
        legend2 = []
    if include_network:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 3))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.2))
    if BigClass.Variabs.task_type != 'Iris_classification':
        ax1.plot(np.linspace(0, t, np.shape(output_in_t)[0]).T,
                 np.asarray(output_in_t)/np.asarray(desired_in_t)-1)
    else:
        ax1.plot(np.linspace(0, t, np.shape(output_in_t)[0]).T,
                 np.asarray(output_in_t))
    ax1.set_title('"measured" output')
    ax1.set_xlabel('t')
    if legend1:
        ax1.legend(legend1)
    ax2.plot(BigClass.State.output_update_in_t[1:])
    ax2.plot(BigClass.State.input_update_in_t[1:])
    if BigClass.Variabs.access_interNodes:
        ax2.plot(BigClass.State.inter_update_in_t[1:])
    ax2.set_title('"update" values')
    ax2.set_xlabel('t')
    if legend2:
        ax2.legend(legend2)
    ax3.plot(BigClass.State.R_in_t[1:])
    ax3.set_title('Resistances')
    ax3.set_xlabel('t')
    # ax3.legend(legend3)
    for t in range(t):
        if t % len(BigClass.Variabs.dataset) == 0 and t != 0 and BigClass.Variabs.task_type != 'Regression':
            ax4.axvline(x=t, color='red', linestyle='--', linewidth=1)
    ax4.plot(BigClass.State.loss_scalar_in_t[1:])
    ax4.set_title('Loss')
    ax4.set_xlabel('t')
    ax4.set_yscale('log')
    if include_network:
        if NET is not None:
            plotNetStructure(NET=BigClass.NET.NET,
                             BigClass=BigClass,
                             pos_lattice=BigClass.NET.pos_lattice,
                             node_labels=node_labels,
                             R_reordered=BigClass.NET.R_reordered,
                             u_reordered=BigClass.NET.u_reordered,
                             p_reordered=BigClass.NET.p_reordered,
                             ax=ax5  # Pass the subplot axis
                             )
        else:
            print('no NET assigned in input')
    plt.show()
