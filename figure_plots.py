from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import copy

from matplotlib.colors import Colormap
from typing import Tuple, List, Dict, Any
from typing import TYPE_CHECKING
from numpy.typing import NDArray
from brokenaxes import brokenaxes

import statistics

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State


# ================================
# functions for paper figure plots
# ================================

## setup params

plt.rcParams['lines.linewidth'] = 2  # Set default line width
plt.rcParams['font.size'] = 14  # Set default font size
plt.rcParams['legend.loc'] = 'best'


## The functions

def loss_afo_in_out(loss_mat: np.ndarray, cmap: Colormap) -> None:
    """
    Nice boxes in cool color scheme of loss a.f.o #inputs and #outputs, lin scale
    use loss_mat outputed from multiple_Nin_Nout.ipynb

    inputs:
    loss_mat: NDArray [Nin, Nout]

    outputs:
    matplotlib figure
    """
    # calculate ensemble mean of loss_mat
    loss_mat_mean = np.mean(loss_mat, axis=2)

    Nin = np.arange(1, np.shape(loss_mat)[0]+1)  # Equivalent to 1:Nin in MATLAB
    Nout = np.arange(1, np.shape(loss_mat)[1]+1)

    # Create the figure and plot
    plt.figure()

    # plot loss_mat without interpolation, setting color limits [0-1]
    plt.imshow(loss_mat_mean, cmap=cmap, origin='lower',
               extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5], vmin=0, vmax=0.3)

    # Labeling
    plt.xlabel('# Outputs')
    plt.ylabel('# Inputs')

    # Set ticks
    plt.xticks(Nin)
    plt.yticks(Nout)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Loss')  # Customize the colorbar label

    # Show the plot
    plt.show()


def plot_performance_2(M: NDArray[np.float_], t: np.int_,
                       output_1in2out: NDArray[np.float_], output_2in1out: NDArray[np.float_],
                       input_dual_1in2out: NDArray[np.float_], input_dual_2in1out: NDArray[np.float_],
                       output_dual_1in2out: NDArray[np.float_], output_dual_2in1out: NDArray[np.float_],
                       R_1in2out: NDArray[np.float_], R_2in1out: NDArray[np.float_],
                       loss_1in2out: NDArray[np.float_], loss_2in1out: NDArray[np.float_],
                       NET_1in2out: nx.DiGraph, NET_2in1out: nx.DiGraph,
                       pos_lattice_1in2out: dict, pos_lattice_2in1out: dict,
                       color_lst: list[str], red: str) -> None:
    """
    one plot with 4 subfigures of
    1) output / desired - 1.
    2) inputs and outputs of the dual problem
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

    # Set the custom color cycle globally without cycler
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', color_lst)

    # sizes for 1 input 2 output
    A_1in2out: float = M[0]  # A = x_hat/p_in
    B_1in2out: float = M[1]  # B = y_hat/p_in
    R_theor_1in2out = np.array([(1-A_1in2out)/(A_1in2out*(1+1)-B_1in2out),
                                (1-B_1in2out)/(B_1in2out*(1+1)-A_1in2out)])
    legend2_1in2out = [r'$x^{\,!}$', r'$y_1^{\,!}$', r'$y_2^{\,!}$']

    # sizes for 2 input 1 output
    A_2in1out = M[0]
    B_2in1out = M[1]
    R_theor_2in1out = np.array([(1-A_2in1out-B_2in1out)/A_2in1out, (1-A_2in1out-B_2in1out)/B_2in1out])
    legend2_2in1out = [r'$x_1^{\,!}$', r'$x_2^{\,!}$', r'$y^{\,!}$']

    # sizes for both
    pos_lattice_both = pos_lattice_2in1out

    # figure
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(17, 6))

    # plt.text(0.5, 0.5, 'matplotlib',
    #          horizontalalignment='center',
    #          verticalalignment='center',
    #          transform = ax.transAxes)

    # plot 1 input 2 output
    ax1.plot(np.mean(np.mean(np.abs(loss_1in2out), axis=1), axis=1))
    ax1.set_yscale('log')
    ax1.set_title(r'$\|\mathcal{L}\|$')
    # ax1.legend(legend4)
    ax2.plot(input_dual_1in2out)
    ax2.plot(output_dual_1in2out)
    ax2.set_title('Dual state pressure')
    ax2.legend(legend2_1in2out, loc='center right')
    ax3.plot(R_1in2out)
    # ax3.plot(np.outer(R_theor_1in2out, np.ones(t)).T, '--')
    ax3.set_title(r'$R$')
    nx.draw_networkx(NET_1in2out, pos=pos_lattice_both, edge_color=color_lst[0], node_color=color_lst[0],
                     with_labels=True, arrows=False, font_color='white', font_size=14, width=2, ax=ax4)
    ax4.set_title('Network structure')

    # plot 2 input 1 output
    ax5.plot(np.mean(np.mean(np.abs(loss_2in1out), axis=1), axis=1))
    ax5.set_xlabel('t')
    ax5.set_yscale('log')
    ax6.plot(input_dual_2in1out)
    ax6.plot(output_dual_2in1out)
    ax6.set_xlabel('t')
    ax6.legend(legend2_2in1out, loc='center right', bbox_to_anchor=(1, 0.4) )
    ax7.plot(R_2in1out)
    # ax8.plot(np.outer(R_theor_2in1out, np.ones(t)).T, '--')
    ax7.set_xlabel('t')
    # ax8.legend(legend3_2in1out)
    nx.draw_networkx(NET_2in1out, pos=pos_lattice_both, edge_color=color_lst[0], node_color=color_lst[0],
                     with_labels=True, arrows=False, font_color='white', font_size=14, width=2, ax=ax8)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        set_thicker_spines(ax)  # Apply the spine thickness to each subplot

    plt.show()


def plot_comparison_pseudo(R_pseudo: NDArray[np.float_], R_network: NDArray[np.float_],
                           loss_pseudo: NDArray[np.float_], loss_network: NDArray[np.float_]) -> None:
    """
    plot comparison of performance of network to those of resistances calculated using
    pseudo inverse method, as in the matlab file "Calculate_desired_resistances_2in3out_theoretical.m"
    one plot with 2 subfigures of
    1) resistances in time
    2) loss in time
    calculated by pseudo inverse (dashed) and network (solid)

    inputs:
    R_pseudo     - resistances calculated using pseudo inverse
    R_network    - resistances of network in time
    loss_pseudo  - loss in time using those found using pseudo inverse (resistances are constant in t)
    loss_network - loss in time using the network (resistances change)
    State   - class instance of the state variables of network
    Variabs - class instance of the variables by the user

    outputs:
    1 matplotlib plot
    """

    # Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    network_color = 'blue'
    pseudo_color = 'violet'
    legend2 = ['Network', 'Pseudo Inverse']

    # Plot resistances in time (ax1)
    network_lines = []
    for i in range(R_network.shape[1]):
        line = ax1.plot(R_network[:, i], color=network_color)
        network_lines.append(line[0])  # Append the first line object from the plot

    # Plot the pseudo inverse line (dashed and violet)
    pseudo_line = ax1.plot(R_pseudo * np.ones([len(R_network), 1]), linestyle='--', color=pseudo_color)[0]

    # Create a custom legend
    # Only take one of the network lines since they all share the same color and appearance
    ax1.legend([network_lines[0], pseudo_line], legend2, loc='best')
    ax1.set_title(r'$R$')

    # Plot loss in time (ax2)
    ax2.plot(np.mean(np.mean(np.abs(loss_network), axis=1), axis=1), label='Network', color=network_color)
    ax2.plot(np.mean(np.mean(np.abs(loss_pseudo), axis=1), axis=1), linestyle='--', label='Pseudo Inverse',
             color=pseudo_color)
    ax2.set_title('|Loss|')
    ax2.set_xlabel('t')
    ax2.set_yscale('log')
    ax2.legend(legend2, loc='best')

    plt.show()


def plot_compare_R_type_loss(Network_1in2out: nx.DiGraph, Network_2in1out: nx.DiGraph,
                             pos_lattice: dict,
                             loss_1in2out_R_propto_deltap: NDArray[np.float_],
                             loss_1in2out_deltaR_propto_deltap: NDArray[np.float_],
                             loss_1in2out_propto_Q: NDArray[np.float_],
                             loss_1in2out_propto_Power: NDArray[np.float_],
                             loss_2in1out_R_propto_deltap: NDArray[np.float_],
                             loss_2in1out_deltaR_propto_deltap: NDArray[np.float_],
                             loss_2in1out_propto_Q: NDArray[np.float_],
                             loss_2in1out_propto_Power: NDArray[np.float_]):
    t = np.shape(loss_1in2out_propto_Power)[0]
    range_vec = range(t)
    range_vec = range(t)
    t_vec = copy.copy(range_vec)
    t_short = 100

    legend = [r'$R \propto \Delta p$',
              r'$\Delta R \propto \Delta p$',
              r'$\Delta R \propto Q$',
              r'$\Delta R \propto \mathrm{Power}$']

    # Initialize broken axes for the top-left plot (ax1)
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2)  # Define 2x2 grid layout

    # Create broken axis in the top-left plot
    baxtop = brokenaxes(xlims=((0, t_short), (t-40, t)), hspace=.05, subplot_spec=gs[0, 0])  # Constrain to top-left

    # Plot data in the broken axis
    baxtop.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_1in2out_R_propto_deltap), axis=1), axis=1)[:t_short],
                label=legend[0])
    baxtop.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_1in2out_deltaR_propto_deltap), axis=1), axis=1)[:t_short],
                label=legend[1])
    baxtop.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_1in2out_propto_Q), axis=1), axis=1)[:t_short],
                label=legend[2])
    baxtop.plot(t_vec[:], np.mean(np.mean(np.abs(loss_1in2out_propto_Power), axis=1), axis=1)[:],
                label=legend[3])

    # Extend the plot from t_short to t_full (the right side of the broken axis)
    # bax.plot(t_vec[t_short:], np.mean(np.mean(np.abs(loss_1in2out_R_propto_deltap), axis=1), axis=1)[t_short:])
    baxtop.set_ylabel(r'$\|\mathcal{L}\|$')

    # Add legend
    baxtop.legend(loc='best')

    # Now create the rest of the figure manually
    ax2 = fig.add_subplot(222)
    nx.draw_networkx(Network_2in1out, pos=pos_lattice, edge_color='b', node_color='b', with_labels=True,
                     font_color='white', font_size=14, ax=ax2)

    # Create broken axis in the top-left plot
    baxbot = brokenaxes(xlims=((0, t_short), (t-40, t)), hspace=.05, subplot_spec=gs[1, 0])  # Constrain to top-left

    # Plot data in the broken axis
    baxbot.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_2in1out_R_propto_deltap), axis=1), axis=1)[:t_short],
                label=legend[0])
    baxbot.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_2in1out_deltaR_propto_deltap), axis=1), axis=1)[:t_short],
                label=legend[1])
    baxbot.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_2in1out_propto_Q), axis=1), axis=1)[:t_short],
                label=legend[2])
    baxbot.plot(t_vec[:], np.mean(np.mean(np.abs(loss_2in1out_propto_Power), axis=1), axis=1)[:],
                label=legend[3])
    baxbot.set_xlabel('t')
    baxbot.set_ylabel(r'$\|\mathcal{L}\|$')
    baxbot.legend(loc='best')

    ax4 = fig.add_subplot(224)
    nx.draw_networkx(Network_1in2out, pos=pos_lattice, edge_color='b', node_color='b', with_labels=True,
                     font_color='white', font_size=14, ax=ax4)

    plt.show()


def plot_accuracy_4_materials(t_final: int, dataset_shape: np.ndarray, t_for_accuracy: np.ndarray,
                              accuracy_in_t_R_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_Q: np.ndarray,
                              accuracy_in_t_deltaR_propto_Power: np.ndarray,
                              colors: List[np.float_], red: str,
                              smooth: bool = True, window_size: int = 5):
    """
    Plots the accuracy in time for the Iris problem using 4 materials.
    """
    # length of classification dataset
    dataset_len = dataset_shape[0]

    opacity = 0.25

    # legend - 4 materials
    legend = [r'$R \propto \Delta p$',
              r'$\Delta R \propto \Delta p$',
              r'$\Delta R \propto Q$',
              r'$\Delta R \propto \mathrm{Power}$']

    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color=red, linestyle='--', linewidth=1)

    # Apply smoothing for the average accuracy lines
    if smooth:
        mean_accuracy_R_propto_deltap = statistics.mov_ave(np.mean(accuracy_in_t_R_propto_deltap, axis=0), window_size)
        mean_accuracy_deltaR_propto_deltap = statistics.mov_ave(np.mean(accuracy_in_t_deltaR_propto_deltap, axis=0),
                                                                window_size)
        mean_accuracy_deltaR_propto_Q = statistics.mov_ave(np.mean(accuracy_in_t_deltaR_propto_Q, axis=0), window_size)
        mean_accuracy_deltaR_propto_Power = statistics.mov_ave(np.mean(accuracy_in_t_deltaR_propto_Power, axis=0),
                                                               window_size)

        # Standard deviations for confidence bounds
        std_R_propto_deltap = statistics.mov_ave(np.std(accuracy_in_t_R_propto_deltap, axis=0), window_size)
        std_deltaR_propto_deltap = statistics.mov_ave(np.std(accuracy_in_t_deltaR_propto_deltap, axis=0), window_size)
        std_deltaR_propto_Q = statistics.mov_ave(np.std(accuracy_in_t_deltaR_propto_Q, axis=0), window_size)
        std_deltaR_propto_Power = statistics.mov_ave(np.std(accuracy_in_t_deltaR_propto_Power, axis=0), window_size)

        t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracy_R_propto_deltap)]  # t_for_accuracy after smoothing
    else:
        mean_accuracy_R_propto_deltap = np.mean(accuracy_in_t_R_propto_deltap, axis=0)
        mean_accuracy_deltaR_propto_deltap = np.mean(accuracy_in_t_deltaR_propto_deltap, axis=0)
        mean_accuracy_deltaR_propto_Q = np.mean(accuracy_in_t_deltaR_propto_Q, axis=0)
        mean_accuracy_deltaR_propto_Power = np.mean(accuracy_in_t_deltaR_propto_Power, axis=0)

        std_R_propto_deltap = np.std(accuracy_in_t_R_propto_deltap, axis=0)
        std_deltaR_propto_deltap = np.std(accuracy_in_t_deltaR_propto_deltap, axis=0)
        std_deltaR_propto_Q = np.std(accuracy_in_t_deltaR_propto_Q, axis=0)
        std_deltaR_propto_Power = np.std(accuracy_in_t_deltaR_propto_Power, axis=0)

        t_for_accuracy_smoothed = t_for_accuracy

    mean_accuracy_R_propto_deltap[0] = 1/3
    mean_accuracy_deltaR_propto_deltap[0] = 1/3
    mean_accuracy_deltaR_propto_Q[0] = 1/3
    mean_accuracy_deltaR_propto_Power[0] = 1/3

    # Plot the smoothed mean accuracy with lines connecting points
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Q,
             color=colors[2], alpha=1., marker=None, linestyle='-', linewidth=3)
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_deltap,
             color=colors[1], alpha=1., marker=None, linestyle='-', linewidth=3)
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Power,
             color=colors[3], alpha=1., marker=None, linestyle='--', linewidth=3)
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_R_propto_deltap,
             color=colors[0], alpha=1., marker=None, linestyle='--', linewidth=3)

    # Plot confidence intervals using fill_between
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Q - std_deltaR_propto_Q,
                     mean_accuracy_deltaR_propto_Q + std_deltaR_propto_Q, color=colors[0], alpha=opacity)
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_deltap - std_deltaR_propto_deltap,
                     mean_accuracy_deltaR_propto_deltap + std_deltaR_propto_deltap, color=colors[1], alpha=opacity)
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Power - std_deltaR_propto_Power,
                     mean_accuracy_deltaR_propto_Power + std_deltaR_propto_Power, color=colors[2], alpha=opacity)
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_R_propto_deltap - std_R_propto_deltap,
                     mean_accuracy_R_propto_deltap + std_R_propto_deltap, color=colors[3], alpha=opacity)

    # Adding a single line for each legend entry with the same colors
    for i in range(4):
        plt.plot([], [], color=colors[i], label=legend[i])

    # axes
    plt.xlabel('t', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(loc='best')
    # plt.xscale('log')
    plt.show()


def plot_accuracy_1_material(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                             dataset_shape: NDArray[np.int_], colors: List[str], red: str,
                             smooth: bool = True, window_size: int = 5) -> None:
    """
    Plots the accuracy in time for the Iris problem

    input:
    t_final        - int, final time step
    t_for_accuracy - array of ints, times during simulation when accuracy was calculated
    accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
    dataset_len    - length of dataset used, for Iris it is 150

    output:
    plot of accuracy a.f.o time
    """
    opacity = 0.25

    # Apply smoothing for the average accuracy lines
    if smooth:
        mean_accuracy = statistics.mov_ave(np.mean(accuracy_in_t, axis=0), window_size)

        # Standard deviations for confidence bounds
        std = statistics.mov_ave(np.std(accuracy_in_t, axis=0), window_size)

        t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracy)]  # t_for_accuracy after smoothing
    else:
        mean_accuracy = np.mean(accuracy_in_t, axis=0)

        std = np.std(accuracy_in_t, axis=0)

        t_for_accuracy_smoothed = t_for_accuracy

    mean_accuracy[0] = 1/3

    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(t_final):
        if t % dataset_shape[0] == 0:
            plt.axvline(x=t, color=red, linestyle='--', linewidth=1)

    # plot accuracy a.f.o time
    plt.plot(t_for_accuracy_smoothed, mean_accuracy, label='accuracy', color=colors[0], marker='.', linestyle='')

    # Plot confidence intervals using fill_between
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy - std,
                     mean_accuracy + std, color=colors[0], alpha=opacity)

    # axes
    plt.xlabel('t', fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Accuracy', fontsize=14)  # Set y-axis label with font size
    # plt.title('Accuracy Over Time', fontsize=16)  # Set title with font size
    plt.ylim([0, 1])


def plot_comparison_R_type(R_propto_deltap: NDArray[np.float_], deltaR_propto_deltap: NDArray[np.float_],
                           deltaR_propto_Q: NDArray[np.float_], deltaR_propto_Power: NDArray[np.float_],
                           loss_R_propto_deltap: NDArray[np.float_],
                           loss_deltaR_propto_deltap: NDArray[np.float_],
                           loss_propto_Q: NDArray[np.float_],
                           loss_propto_Power: NDArray[np.float_]) -> None:
    """
    plot comparison of performance of network to those of resistances calculated using
    pseudo inverse method, as in the matlab file "Calculate_desired_resistances_2in3out_theoretical.m"
    one plot with 2 subfigures of
    1) resistances in time
    2) loss in time
    calculated by pseudo inverse (dashed) and network (solid)

    inputs:
    R_pseudo     - resistances calculated using pseudo inverse
    R_network    - resistances of network in time
    loss_pseudo  - loss in time using those found using pseudo inverse (resistances are constant in t)
    loss_network - loss in time using the network (resistances change)
    State   - class instance of the state variables of network
    Variabs - class instance of the variables by the user

    outputs:
    1 matplotlib plot
    """

    # setups
    fig, axs = plt.subplots(2, 4, figsize=(12, 4))
    (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8) = axs
    R_color = 'blue'
    legend1 = r'$R$'
    legend2 = '|Loss|'

    # Titles for the plots
    titles = [
        r'$R \propto \Delta p$',
        r'$\Delta R \propto \Delta p$',
        r'$R \propto Q$',
        r'$\Delta R \propto \mathrm{Power}$'
    ]

    # Data for the plots
    resistance_data = [R_propto_deltap, deltaR_propto_deltap, deltaR_propto_Q, deltaR_propto_Power]
    loss_data = [loss_R_propto_deltap, loss_deltaR_propto_deltap, loss_propto_Q, loss_propto_Power]

    # Manually set the y-axis sharing for the top row
    for ax in [ax2, ax3, ax4]:
        ax.sharey(ax1)  # Share y-axis with the first subplot (ax1)

    # Plot resistance data (top row)
    for ax, data, title in zip([ax1, ax2, ax3, ax4], resistance_data, titles):
        ax.plot(data)
        ax.set_title(title)
        ax.legend([legend1], loc='best')

    # Plot loss data (bottom row) with independent y-axes
    for ax, data, title in zip([ax5, ax6, ax7, ax8], loss_data, titles):
        ax.plot(np.mean(np.abs(data), axis=1), color=R_color)
        ax.legend([legend2], loc='best')
        ax.set_xlabel('t')
        # ax.set_yscale('log')  # Logarithmic scale, auto-scaled to data

    plt.show()


# Define a function to apply thicker spines globally
def set_thicker_spines(ax, linewidth=2):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
