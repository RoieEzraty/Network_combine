from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import copy

from matplotlib.colors import Colormap
from typing import Tuple, List, Dict, Any
from typing import TYPE_CHECKING
from numpy.typing import NDArray
from brokenaxes import brokenaxes
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import statistics

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State
    from Color_Scheme import Color_Scheme


# ================================
# functions for paper figure plots
# ================================

## setup params

plt.rcParams['lines.linewidth'] = 2  # Set default line width
plt.rcParams['font.size'] = 14  # Set default font size
plt.rcParams['legend.loc'] = 'best'


## The functions


def plot_performance_2(M: NDArray[np.float_], t: np.int_,
                       output_1in2out: NDArray[np.float_], output_2in1out: NDArray[np.float_],
                       input_update_1in2out: NDArray[np.float_], input_update_2in1out: NDArray[np.float_],
                       output_update_1in2out: NDArray[np.float_], output_update_2in1out: NDArray[np.float_],
                       R_1in2out: NDArray[np.float_], R_2in1out: NDArray[np.float_],
                       loss_1in2out: NDArray[np.float_], loss_2in1out: NDArray[np.float_],
                       NET_1in2out: nx.DiGraph, NET_2in1out: nx.DiGraph,
                       pos_lattice_1in2out: dict, pos_lattice_2in1out: dict,
                       Colorscheme: "Color_Scheme") -> None:
    """
    one plot with 4 subfigures of
    1) output / desired - 1.
    2) inputs and outputs of the update modality
    3) resistances in time
    4) absolute mean value of loss in time

    inputs:
    too many

    outputs:
    1 matplotlib plot
    """

    # Set the custom color cycle globally without cycler
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

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
    ax1.set_ylim(None, 1)
    ax1.set_title(r'$\|\mathcal{L}\|$')
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))  # More ticks on the y-axis
    # ax1.legend(legend4)
    ax2.plot(input_update_1in2out)
    ax2.plot(output_update_1in2out)
    ax2.set_title('"Update" modality pressure')
    ax2.legend(legend2_1in2out, loc='center right')
    ax3.plot(R_1in2out)
    # ax3.plot(np.outer(R_theor_1in2out, np.ones(t)).T, '--')
    ax3.set_title(r'$R$')
    nx.draw_networkx(NET_1in2out, pos=pos_lattice_both, edge_color=Colorscheme.colors_lst[0],
                     node_color=Colorscheme.colors_lst[0], with_labels=True, arrows=False, font_color='white',
                     font_size=14, width=2, ax=ax4)
    ax4.set_title('Network structure')

    # plot 2 input 1 output
    ax5.plot(np.mean(np.mean(np.abs(loss_2in1out), axis=1), axis=1))
    ax5.set_xlabel('t')
    ax5.set_yscale('log')
    ax5.set_ylim(None, 1)
    ax6.plot(input_update_2in1out)
    ax6.plot(output_update_2in1out)
    ax6.set_xlabel('t')
    ax6.legend(legend2_2in1out, loc='center right', bbox_to_anchor=(1, 0.4))
    ax7.plot(R_2in1out)
    # ax8.plot(np.outer(R_theor_2in1out, np.ones(t)).T, '--')
    ax7.set_xlabel('t')
    # ax8.legend(legend3_2in1out)
    nx.draw_networkx(NET_2in1out, pos=pos_lattice_both, edge_color=Colorscheme.colors_lst[0],
                     node_color=Colorscheme.colors_lst[0], with_labels=True, arrows=False, font_color='white',
                     font_size=14, width=2, ax=ax8)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        set_thicker_spines(ax)  # Apply the spine thickness to each subplot

    plt.show()


def loss_afo_in_out(loss_mat_lin: np.ndarray, loss_mat_nonlin: np.ndarray, Colorscheme: "Color_Scheme") -> None:
    """
    Two-panel plot comparing linear and nonlinear average loss matrices with an external colorbar.

    Parameters:
    -----------
    loss_mat_lin : np.ndarray
        3D array [Nin, Nout, ...] for the linear system
    loss_mat_nonlin : np.ndarray
        3D array [Nin, Nout, ...] for the nonlinear system
    Colorscheme : Color_Scheme
        Object with a `.cmap` attribute defining the colormap
    """
    loss_mean_lin = np.mean(loss_mat_lin, axis=2)
    loss_mean_nonlin = np.mean(loss_mat_nonlin, axis=2)

    Nin = np.arange(1, loss_mat_lin.shape[0]+1)
    Nout = np.arange(1, loss_mat_lin.shape[1]+1)

    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # cax = fig.add_subplot(gs[2])

    im1 = ax1.imshow(loss_mean_lin, cmap=Colorscheme.cmap, origin='lower',
                     extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5],
                     vmin=0, vmax=0.3)
    ax1.set_title(r'$\dot{R} \propto \Delta p$')
    ax1.set_xlabel('# Outputs')
    ax1.set_ylabel('# Inputs')
    ax1.set_xticks(Nin)
    ax1.set_yticks(Nout)
    set_thicker_spines(ax1, linewidth=1.5)

    im2 = ax2.imshow(loss_mean_nonlin, cmap=Colorscheme.cmap, origin='lower',
                     extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5],
                     vmin=0, vmax=0.3)
    ax2.set_title(r'$\dot{R} \propto \left(\Delta p\right)^3$')
    ax2.set_xlabel('# Outputs')
    ax2.set_xticks(Nin)
    ax2.set_yticks(Nout)
    set_thicker_spines(ax2, linewidth=1.5)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax2)
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cax = divider.append_axes("right", size="5%", pad=0.9)

    cax = ax2.inset_axes((1.05, 0, 0.08, 1.0))
    # Now thicken the colorbar's surrounding box (spines)
    fig.colorbar(im2, cax=cax)

    # Add colorbar to the dedicated axis
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label(r'$\|\mathcal{L}\|$')

    # plt.tight_layout()
    plt.show()


def plot_comparison_GD(R_mine_1in2out: NDArray[np.float_], R_GD_1in2out: NDArray[np.float_],
                       R_mine_2in1out: NDArray[np.float_], R_GD_2in1out: NDArray[np.float_],
                       loss_mine_1in2out: NDArray[np.float_], loss_GD_1in2out: NDArray[np.float_],
                       loss_mine_2in1out: NDArray[np.float_], loss_GD_2in1out: NDArray[np.float_],
                       cosine_sim_1in2out: NDArray[np.float_], cosine_sim_2in1out: NDArray[np.float_],
                       Colorscheme: "Color_Scheme") -> None:

    """
    two rows plot with 3 subfigures each
    1) bar plot of resistances using gradient descent and my scheme
    2) loss in time t using gradient descent and my scheme
    3) cosine similarity between change in conductivities using gradient descent and my scheme

    inputs:
    too many

    outputs:
    1 matplotlib plot
    """

    # Set color cycle globally
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

    # Normalize R values
    R_mine_1in2out_norm = R_mine_1in2out[-1] / np.max(R_mine_1in2out[-1])
    R_GD_1in2out_norm = R_GD_1in2out[-1] / np.max(R_GD_1in2out[-1])
    R_mine_2in1out_norm = R_mine_2in1out[-1] / np.max(R_mine_2in1out[-1])
    R_GD_2in1out_norm = R_GD_2in1out[-1] / np.max(R_GD_2in1out[-1])

    T = 250

    # Average losses
    loss_mine_1in2out_mean = np.mean(np.mean(np.abs(loss_mine_1in2out), axis=1), axis=1)[:T]
    loss_GD_1in2out_mean = np.mean(np.mean(np.abs(loss_GD_1in2out), axis=1), axis=1)[:T]
    loss_mine_2in1out_mean = np.mean(np.mean(np.abs(loss_mine_2in1out), axis=1), axis=1)[:T]
    loss_GD_2in1out_mean = np.mean(np.mean(np.abs(loss_GD_2in1out), axis=1), axis=1)[:T]

    x = np.arange(len(R_GD_1in2out_norm))
    bar_width = 0.35

    # Grid: 2 rows, 3 columns
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.2, 1.2], height_ratios=[1, 1])

    # --- Row 1 ---
    # R bar plot: 1in2out
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x - bar_width / 2, R_GD_1in2out_norm, width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax0.bar(x + bar_width / 2, R_mine_1in2out_norm, width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax0.set_yticks([0, 0.5, 1])
    # ax0.set_ylabel('$R$')
    ax0.set_title('$R$')
    ax0.legend()

    # Loss plot: 1in2out
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(loss_GD_1in2out_mean, label='GD')
    ax1.plot(loss_mine_1in2out_mean, label='this work')
    ax1.set_title(r'$\|\mathcal{L}\|$')
    ax1.set_yscale('log')
    ax1.set_ylim(5e-9, 1)
    # ax1.set_ylabel(r'$\|\mathcal{L}\|$')
    ax1.legend()

    # Cosine similarity: 1in2out
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(cosine_sim_1in2out[:T])
    ax2.plot(np.zeros([T]), '--k')
    # Right panels (cosine similarity)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_title(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
    # ax2.set_ylabel(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
    ax2.set_ylim(-1, 1)

    # --- Row 2 ---
    # R bar plot: 2in1out
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - bar_width / 2, R_GD_2in1out_norm, width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax3.bar(x + bar_width / 2, R_mine_2in1out_norm, width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax3.set_yticks([0, 0.5, 1])
    # ax3.set_ylabel('$R$')
    ax3.set_xlabel('edge #')
    ax3.legend()

    # Loss plot: 2in1out
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(loss_GD_2in1out_mean, label='GD')
    ax4.plot(loss_mine_2in1out_mean, label='this work')
    ax4.set_yscale('log')
    ax4.set_ylim(5e-9, 1)
    # ax4.set_ylabel(r'$\|\mathcal{L}\|$')
    ax4.set_xlabel('$t$')
    ax4.legend()

    # Cosine similarity: 2in1out
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(cosine_sim_2in1out[:T])
    ax5.plot(np.zeros([T]), '--k')
    ax5.set_yticks([-1, 0, 1])
    # ax5.set_ylabel(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
    ax5.set_xlabel('$t$')
    ax5.set_ylim(-1, 1)

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        set_thicker_spines(ax)  # Apply the spine thickness to each subplot

    plt.tight_layout()
    plt.show()


def plot_accuracy_1_material(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                             dataset_shape: NDArray[np.int_], Colorscheme: "Color_Scheme",
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
            plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1)

    # plot accuracy a.f.o time
    plt.plot(t_for_accuracy_smoothed, mean_accuracy, label='accuracy', color=Colorscheme.colors_lst[0], marker='.',
             linestyle='')

    # Plot confidence intervals using fill_between
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy - std,
                     mean_accuracy + std, color=Colorscheme.colors_lst[0], alpha=opacity)

    # axes
    plt.xlabel('$t$', fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Accuracy', fontsize=14)  # Set y-axis label with font size
    # plt.title('Accuracy Over Time', fontsize=16)  # Set title with font size
    plt.ylim([0, 1])

    set_thicker_spines(plt.gca(), linewidth=1.5)  # apply to the current Axes


def plot_accuracy_4_materials(t_final: int, dataset_shape: np.ndarray, t_for_accuracy: np.ndarray,
                              accuracy_in_t_R_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_Q: np.ndarray,
                              accuracy_in_t_deltaR_propto_Power: np.ndarray,
                              Colorscheme: "Color_Scheme", smooth: bool = True, window_size: int = 5):
    """
    Plots the accuracy in time for the Iris problem using 4 materials.
    """
    # length of classification dataset
    dataset_len = dataset_shape[0]

    opacity = 0.25

    # legend - 4 materials
    legend = [r'$R \propto \Delta p$',
              r'$\dot{R} \propto \Delta p$',
              r'$\dot{R} \propto Q$',
              r'$\dot{R} \propto \mathrm{Power}$']

    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1)

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
             color=Colorscheme.colors_lst[2], alpha=1., marker=None, linestyle='-', linewidth=3)
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_deltap,
             color=Colorscheme.colors_lst[1], alpha=1., marker=None, linestyle='-', linewidth=3)
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Power,
             color=Colorscheme.colors_lst[3], alpha=1., marker=None, linestyle='--', linewidth=3)
    plt.plot(t_for_accuracy_smoothed, mean_accuracy_R_propto_deltap,
             color=Colorscheme.colors_lst[0], alpha=1., marker=None, linestyle='--', linewidth=3)

    # Plot confidence intervals using fill_between
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Q - std_deltaR_propto_Q,
                     mean_accuracy_deltaR_propto_Q + std_deltaR_propto_Q, color=Colorscheme.colors_lst[0],
                     alpha=opacity)
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_deltap - std_deltaR_propto_deltap,
                     mean_accuracy_deltaR_propto_deltap + std_deltaR_propto_deltap, color=Colorscheme.colors_lst[1],
                     alpha=opacity)
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_deltaR_propto_Power - std_deltaR_propto_Power,
                     mean_accuracy_deltaR_propto_Power + std_deltaR_propto_Power, color=Colorscheme.colors_lst[2],
                     alpha=opacity)
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy_R_propto_deltap - std_R_propto_deltap,
                     mean_accuracy_R_propto_deltap + std_R_propto_deltap, color=Colorscheme.colors_lst[3],
                     alpha=opacity)

    # Adding a single line for each legend entry with the same colors
    for i in range(4):
        plt.plot([], [], color=Colorscheme.colors_lst[i], label=legend[i])

    # axes
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(loc='best')
    # plt.xscale('log')

    set_thicker_spines(plt.gca(), linewidth=1.5)  # apply to the current Axes

    plt.show()


# Define a function to apply thicker spines globally
def set_thicker_spines(ax, linewidth=2):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


# # NOT IN USE


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
