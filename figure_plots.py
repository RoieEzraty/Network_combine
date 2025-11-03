from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import copy

from typing import Tuple, List, Dict, Any
from typing import TYPE_CHECKING
from numpy.typing import NDArray
from brokenaxes import brokenaxes
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

import statistics

if TYPE_CHECKING:
    from Color_Scheme import Color_Scheme


# ================================
# functions for paper figure plots
# ================================


# # setup params


plt.rcParams['lines.linewidth'] = 2  # Set default line width
plt.rcParams['font.size'] = 14  # Set default font size
plt.rcParams['legend.loc'] = 'best'


# # The functions


def plot_performance_2(M: NDArray[np.float_], t: np.int_,
                       input_update_1in2out: NDArray[np.float_], input_update_2in1out: NDArray[np.float_],
                       output_update_1in2out: NDArray[np.float_], output_update_2in1out: NDArray[np.float_],
                       R_1in2out: NDArray[np.float_], R_2in1out: NDArray[np.float_],
                       loss_1in2out: NDArray[np.float_], loss_2in1out: NDArray[np.float_],
                       NET_1in2out: nx.DiGraph, NET_2in1out: nx.DiGraph,
                       pos_lattice_1in2out: dict, pos_lattice_2in1out: dict,
                       Colorscheme: "Color_Scheme", savestr: str = '') -> None:
    """
    2 rows of 4 subfigures:
    1) Mean Absolute Error a.f.o training time t
    2) inputs and outputs of the update modality
    3) resistances R in time
    4) network structure, using position lattice dictionary pos_lattice
    for the task of 1 input and 2 outputs and 2 inputs 1 output.

    inputs:
    too many

    outputs:
    matplotlib plot
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

    # position lattice is the same for both
    pos_lattice_both = pos_lattice_2in1out

    # custom_labels_1in2out = ['x', 'y₁', 'y₂', r'$⏚$']  # example labels
    # custom_labels_2in1out = ['x₁', 'x₂', 'y', r'$⏚$']  # example labels
    custom_labels_1in2out = [r'$x$', r'$y_1$', r'$y_2$', '']  # example labels
    custom_labels_2in1out = [r'$x_1$', r'$x_2$', r'$y$', '']  # example labels
    # Custom labels dictionary: {node_index: label}
    label_dict_1in2out = {i: custom_labels_1in2out[i] for i in range(len(custom_labels_1in2out))}
    label_dict_2in1out = {i: custom_labels_2in1out[i] for i in range(len(custom_labels_2in1out))}

    arrow_head_w = 28  # arrow head width for network edges

    T = 200  # cutoff time, don't plot after it

    # instantitate figure
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(17, 6))

    # ---- Row 1 - 1 input 2 outputs ----

    # network structure
    # node_colors_1in2out = [Colorscheme.colors_lst[1], Colorscheme.colors_lst[0], Colorscheme.colors_lst[0], 'black']
    node_colors_1in2out = [Colorscheme.colors_lst[0], Colorscheme.colors_lst[1], Colorscheme.colors_lst[2], 'black']
    edge_colors_1in2out = [Colorscheme.colors_lst[0], Colorscheme.colors_lst[1], Colorscheme.colors_lst[2],
                           Colorscheme.colors_lst[3]]
    nx.draw_networkx(NET_1in2out, pos=pos_lattice_both, edge_color=edge_colors_1in2out,
                     node_color=node_colors_1in2out, with_labels=False, arrows=True, font_color='white',
                     font_size=14, width=2, node_size=400, ax=ax1)
    # Add custom labels
    nx.draw_networkx_labels(NET_1in2out, pos=pos_lattice_both, labels=label_dict_1in2out,
                            font_size=16, font_color='white', ax=ax1)
    # Add ground symbols
    add_ground_symbol(ax1, pos_lattice_both, 3)

    ax1.set_title('Network structure')

    # "update" modality pressures
    ax2.plot(input_update_1in2out[1:T])
    ax2.plot(output_update_1in2out[1:T])
    ax2.set_title('Update modality pressure')
    ax2.set_ylim([-0.13, 0.17])
    # ax2.legend(legend2_1in2out, loc='upper right')
    ax2.legend(legend2_1in2out, loc='upper right',
               handletextpad=0.4,    # space between marker and text
               labelspacing=0.3,     # vertical space between labels
               borderaxespad=0.3,     # padding between legend and axes
               )
    # ax2.legend(legend2_1in2out, loc='center right', bbox_to_anchor=(1, 0.4))

    # R
    # ax3.plot(R_1in2out)
    ax3.plot(R_1in2out[:T, :2])
    ax3.plot(R_1in2out[:T, -2:])
    # for theoretical calculation of resistances, not in use
    # ax3.plot(np.outer(R_theor_1in2out, np.ones(t)).T, '--')
    ax3.set_title(r'$R$')

    # ||Loss||
    # ax1.plot(np.mean(np.mean(np.abs(loss_1in2out), axis=1), axis=1))
    ax4.plot(statistics.mov_ave(loss_1in2out[:T], window_size=4))
    ax4.set_yscale('log')
    ax4.set_ylim(1e-15, 1e1)
    ax4.set_title(r'$\|\mathcal{L}\|$')

    # ---- Row 1 - 2 inputs 1 output ----

    # network structure
    node_colors_2in1out = [Colorscheme.colors_lst[0], Colorscheme.colors_lst[2], Colorscheme.colors_lst[1], 'black']
    edge_colors_2in1out = [Colorscheme.colors_lst[0], Colorscheme.colors_lst[2], Colorscheme.colors_lst[1]]
    # Draw arrows for ax5
    # draw_arrow(ax5, pos_lattice_both, 0, 2, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # in to output
    # draw_arrow(ax5, pos_lattice_both, 1, 2, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # in to output
    # draw_arrow(ax5, pos_lattice_both, 2, 3, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # out to ground
    nx.draw_networkx(NET_2in1out, pos=pos_lattice_both, edge_color=edge_colors_2in1out,
                     node_color=node_colors_2in1out, with_labels=False, arrows=True, font_color='white',
                     font_size=14, width=2, node_size=400, ax=ax5)
    # Add custom labels
    nx.draw_networkx_labels(NET_1in2out, pos=pos_lattice_both, labels=label_dict_2in1out,
                            font_size=16, font_color='white', ax=ax5)
    add_ground_symbol(ax5, pos_lattice_both, 3)

    # "update" modality pressures
    ax6.plot(input_update_2in1out[1:T])
    ax6.plot(output_update_2in1out[1:T])
    ax6.set_xlabel('t')
    ax6.set_ylim([-0.13, 0.17])
    # ax6.legend(legend2_2in1out, loc='top right', bbox_to_anchor=(1, 0.4))
    ax6.legend(legend2_2in1out, loc='upper right',
               handletextpad=0.4,    # space between marker and text
               labelspacing=0.3,     # vertical space between labels
               borderaxespad=0.3,     # padding between legend and axes
               )

    # R
    # ax7.plot(R_2in1out)
    ax7.plot(R_2in1out[:T, :2])
    ax7.plot(R_2in1out[:T, -1:])
    # for theoretical calculation of resistances, not in use
    # ax7.plot(np.outer(R_theor_2in1out, np.ones(t)).T, '--')
    ax7.set_xlabel('t')

    # ||Loss||
    ax8.plot(statistics.mov_ave(loss_2in1out[:T], window_size=4))
    ax8.set_xlabel('t')
    ax8.set_yscale('log')
    ax8.set_ylim(1e-6, 1e1)

    fig.subplots_adjust(wspace=0.245)  # widen space between columns

    # Thicker spines
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        set_thicker_spines(ax)  # Apply the spine thickness to each subplot

    if savestr:
        plt.savefig(savestr, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def loss_afo_in_out(loss_mat_lin: np.ndarray, loss_mat_nonlin: np.ndarray, Colorscheme: "Color_Scheme",
                    savestr: str = '') -> None:
    """
    Two-panel plot comparing linear and nonlinear update rules - ensemble mean of loss at end of training,
    shown on a logarithmic color scale.

    Parameters:
    -----------
    loss_mat_lin : np.ndarray
        3D array [Nin, Nout, ensemble] for the linear system
    loss_mat_nonlin : np.ndarray
        3D array [Nin, Nout, ensemble] for the nonlinear system
    Colorscheme : Color_Scheme
        Object with a `.cmap` attribute defining the colormap

    Outputs:
    --------
    matplotlib plot
    """

    log_scale = True
    # log_scale = False
    if log_scale:
        loss_max = 1
        loss_min = 5e-3
        # Set color normalization using logarithmic scale
        norm = LogNorm(vmin=loss_min, vmax=loss_max)  # Adjust vmin/vmax if needed
    else:
        vmin = 0
        vmax = 0.1
        norm = None
    plot_from = 0

    loss_mean_lin = np.mean(loss_mat_lin, axis=2)[plot_from:, plot_from:]
    loss_mean_nonlin = np.mean(loss_mat_nonlin, axis=2)[plot_from:, plot_from:]

    Nin = np.arange(plot_from+1, loss_mat_lin.shape[0]+1)  # array input dimension
    Nout = np.arange(plot_from+1, loss_mat_lin.shape[1]+1)  # array output dimension

    # Instantiate figure and grid for positioning colorbar
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Linear update rule
    ax1.imshow(loss_mean_lin, cmap=Colorscheme.cmap, norm=norm,
               vmin=(vmin if not log_scale else None), vmax=(vmax if not log_scale else None), origin='lower',
               extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5])
    ax1.set_title(r'$\dot{R} \propto \Delta p^{\,!}$')
    ax1.set_xlabel('# Outputs')
    ax1.set_ylabel('# Inputs')
    ax1.set_xticks(Nin)
    ax1.set_yticks(Nout)
    set_thicker_spines(ax1, linewidth=1.5)

    # Nonlinear update rule
    im2 = ax2.imshow(loss_mean_nonlin, cmap=Colorscheme.cmap, norm=norm,
                     vmin=(vmin if not log_scale else None), vmax=(vmax if not log_scale else None), origin='lower',
                     extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5])
    ax2.set_title(r'$\dot{R} \propto \left(\Delta p^{\,!}\right)^3$')
    ax2.set_xlabel('# Outputs')
    ax2.set_xticks(Nin)
    ax2.set_yticks(Nout)
    set_thicker_spines(ax2, linewidth=1.5)

    # Colorbar axis positioned at [1.05, 0] with 0.08 width and 1.0 height
    cax = ax2.inset_axes((1.05, 0, 0.08, 1.0))
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label(r'$\|\mathcal{L}\|$')

    if savestr:
        plt.savefig(savestr, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_accuracy_1_material(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                             dataset_shape: NDArray[np.int_], accuracy_in_t_R_propto_deltap: np.ndarray,
                             accuracy_in_t_deltaR_propto_deltap_nonlin: np.ndarray,
                             accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                             accuracy_in_t_deltaR_propto_Q: np.ndarray,
                             accuracy_in_t_deltaR_propto_Power: np.ndarray, Colorscheme: "Color_Scheme",
                             Iris_PNG_folder: str, smooth: bool = True, window_size: int = 5,
                             savestr: str = '') -> None:
    """
    Plots the accuracy in time for the Iris classification task where R_dot=delta_p

    input:
    t_final        - int, final time step
    t_for_accuracy - array of ints, times during simulation when accuracy was calculated
    accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
    dataset_shape  - shape of dataset used, for Iris it is [150, ?]
    Colorscheme    - Object with a `.cmap` attribute defining the colormap
    smooth         - boolean of whether to perform moveing mean on test accuracy
    window_size    - int, moving mean window

    output:
    plot of accuracy a.f.o time with confidence bounds as STD over ensemble
    """
    opacity = 0.25  # for confidence STD bounds

    T = int(1600/15)  # cutoff time, don't plot after that

    # Apply smoothing for the average accuracy lines
    if smooth:
        mean_accuracy = statistics.mov_ave(np.mean(accuracy_in_t[:, :T], axis=0), window_size)

        # Standard deviations for confidence bounds
        std = statistics.mov_ave(np.std(accuracy_in_t[:, :T], axis=0), window_size)

        t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracy)]  # t_for_accuracy after smoothing
    else:
        mean_accuracy = np.mean(accuracy_in_t[:, :T], axis=0)

        std = np.std(accuracy_in_t[:, :T], axis=0)

        t_for_accuracy_smoothed = t_for_accuracy

    # test accuracy for untrained network is 33%
    mean_accuracy[0] = 1/3

    # plot accuracy a.f.o time
    fig, ax = plt.subplots()

    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(int(t_final)):
        if t % dataset_shape[0] == 0:
            ax.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1, alpha=0.3)

    is_eps = savestr.endswith('.eps')
    opacity_eps = 1.0 if is_eps else opacity

    # Fill confidence bounds FIRST so it's behind the line
    ax.fill_between(t_for_accuracy_smoothed,
                    mean_accuracy - std,
                    mean_accuracy + std,
                    color=Colorscheme.colors_lst[1],
                    alpha=opacity_eps,
                    zorder=1)

    # Then plot the mean accuracy line on top
    ax.plot(t_for_accuracy_smoothed,
            mean_accuracy,
            label='accuracy',
            color=Colorscheme.colors_lst[0],
            marker='.',
            linestyle='',
            zorder=2)

    # ax.plot(t_for_accuracy_smoothed, mean_accuracy, label='accuracy', color=Colorscheme.colors_lst[0], marker='.',
    #         linestyle='')

    # # Plot confidence intervals using fill_between
    # ax.fill_between(t_for_accuracy_smoothed, mean_accuracy - std,
    #                 mean_accuracy + std, color=Colorscheme.colors_lst[0], alpha=opacity)

    # axes
    ax.set_xlabel('$t$', fontsize=14)  # Set x-axis label with font size
    ax.set_xlim([-50, T*15])
    ax.set_ylabel('Test accuracy', fontsize=14)  # Set y-axis label with font size
    ax.set_ylim([0.3, 1])

    # # # Iris
    # # Load your PNG image
    # iris_img = mpimg.imread(Iris_PNG_folder)  # ← path to your image

    # # Create the image box (tweak zoom as needed)
    # imagebox = OffsetImage(iris_img, zoom=0.1)

    # # Anchor it to lower right using axes fraction coordinates
    # ab = AnnotationBbox(imagebox, (0.35, 0.08),  # x, y in axes coords
    #                     frameon=False,
    #                     xycoords='axes fraction',
    #                     box_alignment=(1, 0))  # align bottom-right corner of image to point

    # # Add it to the current axes
    # fig.gca().add_artist(ab)

    # # 5 matrials inset
    # Create inset axes
    inset_ax = ax.inset_axes([0.5, .1, .45, .58])

    # Plot the bar chart into the inset
    plot_final_accuracy_bar_chart(inset_ax,
                                  accuracy_in_t_R_propto_deltap,
                                  accuracy_in_t_deltaR_propto_deltap_nonlin,
                                  accuracy_in_t_deltaR_propto_deltap,
                                  accuracy_in_t_deltaR_propto_Q,
                                  accuracy_in_t_deltaR_propto_Power,
                                  Colorscheme=Colorscheme)

    # Thicker spines
    set_thicker_spines(plt.gca(), linewidth=1.5)  # apply to the current Axes

    if savestr:
        # plt.savefig(savestr, format='eps', dpi=300, bbox_inches='tight')
        plt.savefig(savestr, format='eps', dpi=300)
    else:
        plt.show()


def plot_final_accuracy_bar_chart(ax: plt.Axes,
                                  accuracy_in_t_R_propto_deltap: np.ndarray,
                                  accuracy_in_t_deltaR_propto_deltap_nonlin: np.ndarray,
                                  accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                                  accuracy_in_t_deltaR_propto_Q: np.ndarray,
                                  accuracy_in_t_deltaR_propto_Power: np.ndarray,
                                  Colorscheme: "Color_Scheme") -> None:
    """
    Plots a bar chart of final average test accuracy into a provided Axes object.
    """
    legend = [r'$\dot{R} \propto \Delta p$',
              r'$\dot{R} \propto \left(\Delta p\right)^3$',
              r'$R \propto \Delta p$',
              r'$\dot{R} \propto Q$',
              r'$\dot{R} \propto \Pi$']
    accuracy_data = [accuracy_in_t_deltaR_propto_deltap,
                     accuracy_in_t_deltaR_propto_deltap_nonlin,
                     accuracy_in_t_R_propto_deltap,
                     accuracy_in_t_deltaR_propto_Q,
                     accuracy_in_t_deltaR_propto_Power]

    means = []
    stds = []

    for data in accuracy_data:
        final_46 = data[:, -46:]
        mean_per_trial = np.mean(final_46, axis=1)
        means.append(np.mean(mean_per_trial))
        stds.append(np.std(mean_per_trial))

    x = np.arange(len(legend))
    colors = Colorscheme.colors_lst[:len(legend)]

    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
    # ax.set_xticks(x)
    # ax.set_xticklabels(legend, rotation=20, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Hide default labels

    # Add rotated labels inside the bars
    for i, label in enumerate(legend):
        ax.text(x[i], 0.35, label, ha='center', va='bottom', rotation=90, fontsize=13)
    ax.set_ylabel("Final accuracy", fontsize=14)
    ax.set_ylim([0.33, 1])
    ax.set_yticks(np.arange(0.3, 1.01, 0.2))
    ax.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.4)

    set_thicker_spines(ax, linewidth=1.0)


# # APPENDICES


def comparison_hidden_layers(Nin1_Nout7_Ninter0_lin: np.ndarray, Nin1_Nout7_Ninter7_lin: np.ndarray,
                             Nin1_Nout7_Ninter0_nonlin: np.ndarray, Nin1_Nout7_Ninter7_nonlin: np.ndarray,
                             Nin6_Nout1_Ninter0_lin: np.ndarray, Nin6_Nout1_Ninter6_lin: np.ndarray,
                             Nin6_Nout1_Ninter0_nonlin: np.ndarray, Nin6_Nout1_Ninter6_nonlin: np.ndarray,
                             Nin6_Nout7_Ninter0_lin: np.ndarray, Nin6_Nout7_Ninter7_lin: np.ndarray,
                             Nin6_Nout7_Ninter0_nonlin: np.ndarray, Nin6_Nout7_Ninter7_nonlin: np.ndarray,
                             Colorscheme: "Color_Scheme", savestr: str = '') -> None:

    window_size = 100

    # loss_Nin1_Nout7_Ninter0_lin = statistics.mov_ave(Nin1_Nout7_Ninter0_lin, window_size)[-1]
    # loss_Nin1_Nout7_Ninter7_lin = statistics.mov_ave(Nin1_Nout7_Ninter7_lin, window_size)[-1]
    # loss_Nin1_Nout7_Ninter0_nonlin = statistics.mov_ave(Nin1_Nout7_Ninter0_nonlin, window_size)[-1]
    # loss_Nin1_Nout7_Ninter7_nonlin = statistics.mov_ave(Nin1_Nout7_Ninter7_nonlin, window_size)[-1]

    # loss_Nin6_Nout1_Ninter0_lin = statistics.mov_ave(Nin6_Nout1_Ninter0_lin, window_size)[-1]
    # loss_Nin6_Nout1_Ninter6_lin = statistics.mov_ave(Nin6_Nout1_Ninter6_lin, window_size)[-1]
    # loss_Nin6_Nout1_Ninter0_nonlin = statistics.mov_ave(Nin6_Nout1_Ninter0_nonlin, window_size)[-1]
    # loss_Nin6_Nout1_Ninter6_nonlin = statistics.mov_ave(Nin6_Nout1_Ninter6_nonlin, window_size)[-1]

    loss_Nin6_Nout7_Ninter0_lin = statistics.mov_ave(Nin6_Nout7_Ninter0_lin, window_size)[-1]
    loss_Nin6_Nout7_Ninter7_lin = statistics.mov_ave(Nin6_Nout7_Ninter7_lin, window_size)[-1]
    loss_Nin6_Nout7_Ninter0_nonlin = statistics.mov_ave(Nin6_Nout7_Ninter0_nonlin, window_size)[-1]
    loss_Nin6_Nout7_Ninter7_nonlin = statistics.mov_ave(Nin6_Nout7_Ninter7_nonlin, window_size)[-1]

    # std_Nin1_Nout7_Ninter0_lin = np.std(Nin1_Nout7_Ninter0_lin[-window_size:])
    # std_Nin1_Nout7_Ninter7_lin = np.std(Nin1_Nout7_Ninter7_lin[-window_size:])
    # std_Nin1_Nout7_Ninter0_nonlin = np.std(Nin1_Nout7_Ninter0_nonlin[-window_size:])
    # std_Nin1_Nout7_Ninter7_nonlin = np.std(Nin1_Nout7_Ninter7_nonlin[-window_size:])

    # std_Nin6_Nout1_Ninter0_lin = np.std(Nin6_Nout1_Ninter0_lin[-window_size:])
    # std_Nin6_Nout1_Ninter6_lin = np.std(Nin6_Nout1_Ninter6_lin[-window_size:])
    # std_Nin6_Nout1_Ninter0_nonlin = np.std(Nin6_Nout1_Ninter0_nonlin[-window_size:])
    # std_Nin6_Nout1_Ninter6_nonlin = np.std(Nin6_Nout1_Ninter6_nonlin[-window_size:])

    std_Nin6_Nout7_Ninter0_lin = np.std(Nin6_Nout7_Ninter0_lin[-window_size:])
    std_Nin6_Nout7_Ninter7_lin = np.std(Nin6_Nout7_Ninter7_lin[-window_size:])
    std_Nin6_Nout7_Ninter0_nonlin = np.std(Nin6_Nout7_Ninter0_nonlin[-window_size:])
    std_Nin6_Nout7_Ninter7_nonlin = np.std(Nin6_Nout7_Ninter7_nonlin[-window_size:])

    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 0.05], hspace=0.05, wspace=0.1)

    # Upper and lower axes for the broken y-axis on the left subplot
    # ax1_upper = fig.add_subplot(gs[0, 0])
    # ax1_lower = fig.add_subplot(gs[1, 0], sharex=ax1_upper)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1], sharey=ax1)  # full height

    # Hide x-axis ticks on the upper part
    # ax1_upper.tick_params(labelbottom=False)
    ax1.tick_params(bottom=False, labelbottom=False)
    ax2.tick_params(bottom=False, labelbottom=False, labelleft=False)

    # x_labels = [r'$1\ \mathit{in}\ 7\ \mathit{out}$', r'$6\ \mathit{in}\ 1\ \mathit{out}$',
    #             r'$6\ \mathit{in}\ 7\ \mathit{out}$']
    bar_positions = np.arange(1)
    bar_width = 0.35

    # Data
    # means_no_hidden = [loss_Nin1_Nout7_Ninter0_lin, loss_Nin6_Nout1_Ninter0_lin, loss_Nin6_Nout7_Ninter0_lin]
    # means_with_hidden = [loss_Nin1_Nout7_Ninter7_lin, loss_Nin6_Nout1_Ninter6_lin, loss_Nin6_Nout7_Ninter7_lin]

    # stds_no_hidden = [std_Nin1_Nout7_Ninter0_lin, std_Nin6_Nout1_Ninter0_lin, std_Nin6_Nout7_Ninter0_lin]
    # stds_with_hidden = [std_Nin1_Nout7_Ninter7_lin, std_Nin6_Nout1_Ninter6_lin, std_Nin6_Nout7_Ninter7_lin]

    # # Plot on both upper and lower axes
    # for ax in [ax1_upper, ax1_lower]:
    #     ax.bar(bar_positions - bar_width/2, means_no_hidden, bar_width,
    #            yerr=stds_no_hidden, capsize=5, edgecolor='k', linewidth=1.6, label='no hidden')
    #     ax.bar(bar_positions + bar_width/2, means_with_hidden, bar_width,
    #            yerr=stds_with_hidden, capsize=5, edgecolor='k', linewidth=1.6, label='with hidden')

    ax1.bar(bar_positions - bar_width/2,
            loss_Nin6_Nout7_Ninter0_lin,
            bar_width,
            yerr=std_Nin6_Nout7_Ninter0_lin,
            capsize=5, edgecolor='k', linewidth=1.6, label='no hidden')

    ax1.bar(bar_positions + bar_width/2,
            loss_Nin6_Nout7_Ninter7_lin,
            bar_width,
            yerr=std_Nin6_Nout7_Ninter7_lin,
            capsize=5, edgecolor='k', linewidth=1.6, label='with hidden')

    ax2.bar(bar_positions - bar_width/2,
            loss_Nin6_Nout7_Ninter0_nonlin,
            bar_width,
            yerr=std_Nin6_Nout7_Ninter0_nonlin,
            capsize=5, edgecolor='k', linewidth=1.6, label='no hidden')

    ax2.bar(bar_positions + bar_width/2,
            loss_Nin6_Nout7_Ninter7_nonlin,
            bar_width,
            yerr=std_Nin6_Nout7_Ninter7_nonlin,
            capsize=5, edgecolor='k', linewidth=1.6, label='with hidden')

    # ax1_upper.set_ylim(1e-6, 1)    # upper part
    # ax1_lower.set_ylim(1e-21, 5e-20)  # lower part
    # ax1_upper.set_yscale('log')
    # ax1_lower.set_yscale('log')

    # d = .015  # size of diagonal lines
    # kwargs = dict(transform=ax1_upper.transAxes, color='k', clip_on=False)
    # ax1_upper.plot((-d, +d), (-d, +d), **kwargs)        # top-left
    # ax1_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right

    # kwargs.update(transform=ax1_lower.transAxes)  # switch to the bottom axes
    # ax1_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left
    # ax1_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right

    # ax1_lower.set_xticks(bar_positions)
    # ax1_lower.set_xticklabels(x_labels, rotation=35)
    # ax1_upper.set_title(r'$\dot{R}\propto\Delta p^{\,!}$')
    # ax1_upper.set_ylabel(r'$\|\mathcal{L}\|$')

    ax1.set_title(r'$\dot{R}\propto\Delta p^{\,!}$')
    # ax1.set_xticks(bar_positions)
    # ax1.set_xticklabels(x_labels, rotation=35)
    ax1.set_ylabel(r'$\|\mathcal{L}\|$')
    ax1.set_yscale('log')
    ax1.set_ylim([1e-3, 1])
    # ax1.legend()

    ax2.set_title(r'$\dot{R}\propto\left(\Delta p^{\,!}\right)^3$')
    # ax2.set_xticks(bar_positions)
    # ax2.set_xticklabels(x_labels, rotation=35)
    # ax2.set_ylabel(r'$\|\mathcal{L}\|$')
    ax2.set_yscale('log')
    ax2.set_ylim([1e-3, 1])
    ax2.legend()

    # ax1.set_title(r'$\dot{R}\propto\Delta p^{\,!}$')
    # ax1.set_xticks(bar_positions)
    # ax1.set_xticklabels(x_labels, rotation=90)
    # ax1.set_ylabel(r'$\|\mathcal{L}\|$')
    # ax1.set_ylim([1e-8,1])
    # # ax1.set_ylim([0,0.06])
    # ax1.set_yscale('log')
    # # ax1.legend()

    # ax2.set_title(r'$\dot{R}\propto\left(\Delta p^{\,!}\right)^3$')
    # ax2.set_xticks(bar_positions)
    # ax2.set_xticklabels(x_labels, rotation=90)
    # # ax2.set_ylabel(r'$\|\mathcal{L}\|$')
    # ax2.set_ylim([1e-6,1])
    # # ax2.set_ylim([0,0.06])
    # ax2.set_yscale('log')
    # ax2.legend()

    plt.tight_layout()

    if savestr:
        plt.savefig(savestr, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_comparison_GD_4in6out(R_Adalike_4in6out: NDArray[np.float_], R_GD_4in6out: NDArray[np.float_], Nout: int,
                               cosine_sim_4in6out: NDArray[np.float_], Colorscheme: "Color_Scheme",
                               window: int = 0, savestr: str = '') -> None:

    """
    1) Bar plot of resistances at end of training using gradient descent (GD) and proposed scheme
    2) cosine similarity between change in conductivities using GD and my scheme

    inputs:
    too many

    outputs:
    matplotlib plot
    """
    # Set color cycle globally
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

    # omit output to ground
    R_Adalike_4in6out = R_Adalike_4in6out[-1][:-Nout]
    R_GD_4in6out = R_GD_4in6out[-1][:-Nout]

    # Normalize R values so maximal will be 1
    # R_Adalike_4in6out_norm = R_Adalike_4in6out / np.max(R_Adalike_4in6out)
    # R_GD_4in6out_norm = R_GD_4in6out / np.max(R_GD_4in6out)
    R_Adalike_4in6out_norm = R_Adalike_4in6out
    R_GD_4in6out_norm = R_GD_4in6out

    # # weird setup for bars
    # x_4in6out = np.arange(len(R_GD_4in6out_norm))
    # bar_width = 0.35

    if window:
        cosine_sim_4in6out = statistics.mov_ave(cosine_sim_4in6out, window)

    # Grid: 1 rows, 2 columns
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # R vs R_GD
    # ax0.bar(x_4in6out - bar_width / 2, R_GD_4in6out_norm,
    #         width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
    # ax0.bar(x_4in6out + bar_width / 2, R_Adalike_4in6out_norm,
    #         width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
    # ax0.set_yticks([0, 0.5, 1])
    # ax0.set_xticks([0, 1, 2, 3])
    # ax0.plot(R_GD_4in6out_norm, R_Adalike_4in6out_norm, '.')

    # color_cycle = Colorscheme.colors_lst + ['#000000'] + Colorscheme.colors_lst
    color_cycle = [Colorscheme.colors_lst[0]]*len(R_GD_4in6out)
    print('color_cycle', color_cycle)
    n_colors = Nout

    # Assign color by position modulo color count
    color_indices = np.arange(len(R_GD_4in6out_norm)) % n_colors
    point_colors = [color_cycle[i] for i in color_indices]

    print('point colors', point_colors)
    # Scatter plot with per-point colors
    ax0.scatter(
        R_GD_4in6out_norm,
        R_Adalike_4in6out_norm,
        c=point_colors,
        marker='.',
        s=40,
        edgecolors='none'
    )
    ax0.set_ylabel(r'$R$')
    ax0.set_xlabel(r'$R_{GD}$')

    # Diagonal line (y = x)
    min_val = min(R_GD_4in6out_norm.min(), R_Adalike_4in6out_norm.min())
    max_val = max(R_GD_4in6out_norm.max(), R_Adalike_4in6out_norm.max())
    ax0.plot([min_val, max_val], [min_val, max_val], '--k', linewidth=1.2, alpha=0.3)

    # Set log scale
    ax0.set_xscale('log')
    ax0.set_yscale('log')

    # Cosine similarity
    ax1.plot(cosine_sim_4in6out)
    # ax1.plot(np.zeros([len(cosine_sim_4in6out)]), '--k')  # dotted line at cosine=0
    ax1.set_ylabel(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
    ax1.set_xlabel('$t$')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.5, 1])

    # Thicker spines
    for ax in [ax0, ax1]:
        set_thicker_spines(ax)

    plt.tight_layout()

    if savestr:
        plt.savefig(savestr, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def cos_sim_lin_nonlin(cos_mat_lin: np.ndarray, cos_mat_nonlin: np.ndarray, Colorscheme: "Color_Scheme",
                       savestr: str = '') -> None:
    """
    Two-panel plot comparing linear and nonlinear update rules - ensemble mean of loss at end of training,
    shown on a logarithmic color scale.

    Parameters:
    -----------
    loss_mat_lin : np.ndarray
        3D array [Nin, Nout, ensemble] for the linear system
    loss_mat_nonlin : np.ndarray
        3D array [Nin, Nout, ensemble] for the nonlinear system
    Colorscheme : Color_Scheme
        Object with a `.cmap` attribute defining the colormap

    Outputs:
    --------
    matplotlib plot
    """

    vmin = 0
    vmax = 1

    cos_mean_lin = np.mean(cos_mat_lin, axis=2)
    cos_mean_nonlin = np.mean(cos_mat_nonlin, axis=2)

    Nin = np.arange(1, 11)  # array input dimension
    Nout = np.arange(1, 11)  # array output dimension

    # Instantiate figure and grid for positioning colorbar
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Linear update rule
    ax1.imshow(cos_mean_lin, cmap=Colorscheme.cmap, vmin=vmin, vmax=vmax, origin='lower',
               extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5])
    ax1.set_title(r'$\dot{R} \propto \Delta p$')
    ax1.set_xlabel('# Outputs')
    ax1.set_ylabel('# Inputs')
    ax1.set_xticks(Nin)
    ax1.set_yticks(Nout)
    set_thicker_spines(ax1, linewidth=1.5)

    # Nonlinear update rule
    im2 = ax2.imshow(cos_mean_nonlin, cmap=Colorscheme.cmap, vmin=vmin, vmax=vmax, origin='lower',
                     extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5])
    ax2.set_title(r'$\dot{R} \propto \left(\Delta p\right)^3$')
    ax2.set_xlabel('# Outputs')
    ax2.set_xticks(Nin)
    ax2.set_yticks(Nout)
    set_thicker_spines(ax2, linewidth=1.5)

    # Colorbar axis positioned at [1.05, 0] with 0.08 width and 1.0 height
    cax = ax2.inset_axes((1.05, 0, 0.08, 1.0))
    cbar = fig.colorbar(im2, cax=cax)
    # cbar.set_label(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
    cbar.set_label(r'$C$')

    if savestr:
        plt.savefig(savestr, format='eps', dpi=300, bbox_inches='tight')
    else:
        plt.show()


# # Auxiliary functions


# Apply thicker spines globally
def set_thicker_spines(ax, linewidth=2):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


# Add a ground symbol beside a node in network structure
def add_ground_symbol(ax, pos, node_index):
    """
    Add a ground symbol offset to the left and below the node at pos[node_index],
    connected with two lines: one left, then one down.
    """
    x, y = pos[node_index]
    x_offset = -0.25  # shift left
    y_offset = -0.25  # shift down

    corner_x = x + x_offset
    corner_y = y

    ground_x = corner_x
    ground_y = y + y_offset

    # Horizontal line left
    ax.add_line(Line2D([x, corner_x], [y, corner_y], color='black', linewidth=2))
    # Vertical line down
    ax.add_line(Line2D([corner_x, ground_x], [corner_y, ground_y + 2.5*0.045], color='black', linewidth=2))

    # Add ground symbol: 3 horizontal lines decreasing in width
    for i, width in enumerate([0.06, 0.12, 0.18]):
        ax.add_line(Line2D([ground_x - width / 2, ground_x + width / 2],
                           [ground_y + i * 0.045, ground_y + i * 0.045],
                           color='black', linewidth=1.5))


# draw thicker arrows in network structure
def draw_arrow(ax, pos, src, dst, color='gray', arrowstyle='-|>', lw=2, head_width=6):
    """Draws a thick arrow from node `src` to node `dst` on axes `ax`."""
    src_xy = pos[src]
    dst_xy = pos[dst]
    eps = 0
    arrow = FancyArrowPatch(
        posA=src_xy+eps,
        posB=dst_xy+eps,
        connectionstyle="arc3,rad=0.0",
        arrowstyle=arrowstyle,
        mutation_scale=head_width,  # size of the arrow head
        linewidth=lw,
        color=color,
        zorder=1
    )
    ax.add_patch(arrow)


# # NOT IN USE


# def plot_accuracy_5_materials(t_final: int, dataset_shape: np.ndarray, t_for_accuracy: np.ndarray,
#                               accuracy_in_t_R_propto_deltap: np.ndarray,
#                               accuracy_in_t_deltaR_propto_deltap_nonlin: np.ndarray,
#                               accuracy_in_t_deltaR_propto_deltap: np.ndarray,
#                               accuracy_in_t_deltaR_propto_Q: np.ndarray,
#                               accuracy_in_t_deltaR_propto_Power: np.ndarray,
#                               Colorscheme: "Color_Scheme", smooth: bool = True, window_size: int = 5):
#     """
#     Plots the accuracy in time for the Iris classification task using 4 materials.

#     input:
#     t_final        - int, final time step
#     t_for_accuracy - array of ints, times during simulation when accuracy was calculated
#     accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
#     dataset_shape  - shape of dataset used, for Iris it is [150, ?]
#     Colorscheme    - Object with a `.cmap` attribute defining the colormap
#     smooth         - boolean of whether to perform moving mean on test accuracy
#     window_size    - int, moving mean window

#     output:
#     plot of accuracy a.f.o time with confidence bounds as STD over ensemble
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     dataset_len = dataset_shape[0]
#     opacity = 0.25

#     material_keys = ['deltaR_deltap', 'deltaR_deltap_nonlin', 'deltaR_Q', 'deltaR_Power', 'R_deltap']
#     legend = [r'$\dot{R} \propto \Delta p$',
#               r'$\dot{R} \propto \left(\Delta p\right)^3$',
#               r'$\dot{R} \propto Q$',
#               r'$\dot{R} \propto \mathrm{Power}$',
#               r'$R \propto \Delta p$']
#     accuracy_data = [accuracy_in_t_deltaR_propto_deltap,
#                      accuracy_in_t_deltaR_propto_deltap_nonlin,
#                      accuracy_in_t_deltaR_propto_Q,
#                      accuracy_in_t_deltaR_propto_Power,
#                      accuracy_in_t_R_propto_deltap]

#     mean_accuracies = {}
#     std_accuracies = {}

#     for key, data in zip(material_keys, accuracy_data):
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#         if smooth:
#             mean = statistics.mov_ave(mean, window_size)
#             std = statistics.mov_ave(std, window_size)
#         mean_accuracies[key] = mean
#         std_accuracies[key] = std/2

#     if smooth:
#         t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracies['R_deltap'])]
#     else:
#         t_for_accuracy_smoothed = t_for_accuracy

#     # test accuracy for untrained network is 33%
#     for key in material_keys:
#         mean_accuracies[key][0] = 1 / 3

#     # Vertical lines to indicate dataset cycles
#     for t in range(t_final):
#         if t % dataset_len == 0:
#             plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1)

#     # Plotting
#     line_styles = ['-', '-', '-', '--', '--']
#     for i, key in enumerate(material_keys):
#         plt.plot(t_for_accuracy_smoothed, mean_accuracies[key],
#                  color=Colorscheme.colors_lst[i],
#                  linestyle=line_styles[i], linewidth=3, alpha=1., marker=None)

#     for i, key in enumerate(material_keys):
#         plt.fill_between(t_for_accuracy_smoothed,
#                          mean_accuracies[key] - std_accuracies[key],
#                          mean_accuracies[key] + std_accuracies[key],
#                          color=Colorscheme.colors_lst[i], alpha=opacity)

#     for i in range(4):
#         plt.plot([], [], color=Colorscheme.colors_lst[i], label=legend[i])

#     # axes
#     plt.xlabel('$t$', fontsize=14)
#     plt.ylabel('Test accuracy', fontsize=14)
#     plt.ylim([0, 1])
#     plt.legend(loc='best')

#     set_thicker_spines(plt.gca(), linewidth=1.5)
#     plt.show()



# def plot_comparison_Adaline(R_3in1out_Adaline: NDArray[np.float_],
#                             R_3in1out_ourscheme: NDArray[np.float_],
#                             loss_3in1out_Adaline: NDArray[np.float_],
#                             loss_3in1out_ourscheme: NDArray[np.float_],
#                             Colorscheme: "Color_Scheme", smooth: bool = True,
#                             window_size: int = 10) -> None:

#     # Set color cycle globally
#     plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

#     # Normalize R values so maximal will be 1
#     R_3in1out_Adaline_norm = R_3in1out_Adaline[-1] / np.max(R_3in1out_Adaline[-1])
#     R_3in1out_ourscheme_norm = R_3in1out_ourscheme[-1] / np.max(R_3in1out_ourscheme[-1])

#     R_3in1out_Adaline_norm = np.concatenate([R_3in1out_Adaline_norm[:3], R_3in1out_Adaline_norm[-1:]])
#     R_3in1out_ourscheme_norm = np.concatenate([R_3in1out_ourscheme[:3],  R_3in1out_ourscheme[-1:]])

#     # only use run up to t=5600
#     T = 6000

#     # MAE Loss up to t=T
#     loss_3in1out_Adaline_norm = np.mean(np.mean(np.abs(loss_3in1out_Adaline), axis=1), axis=1)[:T]
#     loss_3in1out_ourscheme_norm = np.mean(np.mean(np.abs(loss_3in1out_ourscheme), axis=1), axis=1)[:T]

#     if smooth:
#         loss_3in1out_Adaline_norm = statistics.mov_ave(loss_3in1out_Adaline_norm, window_size)
#         loss_3in1out_ourscheme_norm = statistics.mov_ave(loss_3in1out_ourscheme_norm, window_size)

#     # weird setup for bars
#     x = np.arange(len(R_3in1out_ourscheme_norm))
#     bar_width = 0.35

#     # instantiate figure and grid for positioning colorbal
#     fig = plt.figure(figsize=(7, 3))
#     gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.45)

#     # R bar plot
#     ax0 = fig.add_subplot(gs[0, 0])
#     ax0.bar(x - bar_width / 2, R_3in1out_Adaline_norm,
#             width=bar_width, label='Adaline-like', alpha=0.8, edgecolor='k', linewidth=1.6)
#     ax0.bar(x + bar_width / 2, R_3in1out_Adaline_norm,
#             width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
#     ax0.set_xticks([0, 1, 2, 3])
#     ax0.set_yticks([0, 0.5, 1])
#     ax0.set_ylabel('$R$')
#     # ax0.legend()

#     # Loss plot
#     ax1 = fig.add_subplot(gs[0, 1])
#     ax1.plot(loss_3in1out_Adaline_norm, label='Adaline-like')
#     ax1.plot(loss_3in1out_ourscheme_norm, label='this work')
#     ax1.set_ylabel(r'$\|\mathcal{L}\|$')
#     ax1.set_xlabel(r'$t$')
#     ax1.set_yscale('log')
#     ax1.set_ylim(1e-3, 1)
#     ax1.legend()

#     # Thicker spines
#     for ax in [ax0, ax1]:
#         set_thicker_spines(ax)

#     plt.tight_layout()
#     plt.show()


# def plot_comparison_GD(R_mine_1in2out: NDArray[np.float_], R_GD_1in2out: NDArray[np.float_],
#                        R_mine_2in1out: NDArray[np.float_], R_GD_2in1out: NDArray[np.float_],
#                        cosine_sim_1in2out: NDArray[np.float_], cosine_sim_2in1out: NDArray[np.float_],
#                        Colorscheme: "Color_Scheme") -> None:

#     """
#     Two rows plot with 3 subfigures each
#     1) Bar plot of resistances at end of training using gradient descent (GD) and proposed scheme
#     2) loss a.f.o t using GD and proposed scheme
#     3) cosine similarity between change in conductivities using GD and my scheme

#     inputs:
#     too many

#     outputs:
#     matplotlib plot
#     """
#     # Set color cycle globally
#     plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

#     # Normalize R values so maximal will be 1
#     R_mine_1in2out_norm = R_mine_1in2out[-1] / np.max(R_mine_1in2out[-1])
#     R_GD_1in2out_norm = R_GD_1in2out[-1] / np.max(R_GD_1in2out[-1])
#     R_mine_2in1out_norm = R_mine_2in1out[-1] / np.max(R_mine_2in1out[-1])
#     R_GD_2in1out_norm = R_GD_2in1out[-1] / np.max(R_GD_2in1out[-1])

#     # omit input to ground
#     R_mine_1in2out_norm = np.concatenate([R_mine_1in2out_norm[:2], R_mine_1in2out_norm[-2:]])
#     R_GD_1in2out_norm = np.concatenate([R_GD_1in2out_norm[:2], R_GD_1in2out_norm[-2:]])
#     R_mine_2in1out_norm = np.concatenate([R_mine_2in1out_norm[:2], R_mine_2in1out_norm[-1:]])
#     R_GD_2in1out_norm = np.concatenate([R_GD_2in1out_norm[:2], R_GD_2in1out_norm[-1:]])

#     # weird setup for bars
#     x_1in2out = np.arange(len(R_GD_1in2out_norm))
#     x_2in1out = np.arange(len(R_GD_2in1out_norm))
#     bar_width = 0.35

#     # Grid: 2 rows, 3 columns
#     fig = plt.figure(figsize=(15, 6))
#     gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.2], height_ratios=[1, 1])

#     # ---- Row 1 - 1 input 2 outputs ----

#     # R bar plot
#     ax0 = fig.add_subplot(gs[0, 0])
#     ax0.bar(x_1in2out - bar_width / 2, R_GD_1in2out_norm,
#             width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
#     ax0.bar(x_1in2out + bar_width / 2, R_mine_1in2out_norm,
#             width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
#     ax0.set_yticks([0, 0.5, 1])
#     ax0.set_xticks([0, 1, 2, 3])
#     ax0.set_title('$R$')
#     ax0.legend()

#     # Cosine similarity
#     ax1 = fig.add_subplot(gs[0, 1])
#     ax1.plot(cosine_sim_1in2out)
#     ax1.plot(np.zeros([len(cosine_sim_1in2out)]), '--k')  # dotted line at cosine=0
#     ax1.set_yticks([-1, 0, 1])
#     ax1.set_title(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
#     ax1.set_ylim(-1, 1)

#     # ---- Row 1 - 2 inputs 1 output ----

#     # R bar plot
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.bar(x_2in1out - bar_width / 2, R_GD_2in1out_norm,
#             width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
#     ax2.bar(x_2in1out + bar_width / 2, R_mine_2in1out_norm,
#             width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
#     ax2.set_xticks([0, 1, 2,])
#     ax2.set_yticks([0, 0.5, 1])
#     ax2.set_xlabel('edge #')
#     ax2.legend()

#     # Cosine similarity
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.plot(cosine_sim_2in1out)
#     ax3.plot(np.zeros([len(cosine_sim_2in1out)]), '--k')  # dotted line at cosine=0
#     ax3.set_yticks([-1, 0, 1])
#     ax3.set_xlabel('$t$')
#     ax3.set_ylim(-1, 1)

#     # Thicker spines
#     for ax in [ax0, ax1, ax2, ax3]:
#         set_thicker_spines(ax)

#     plt.tight_layout()
#     plt.show()


# def plot_accuracy_4_materials(t_final: int, dataset_shape: np.ndarray, t_for_accuracy: np.ndarray,
#                               accuracy_in_t_R_propto_deltap: np.ndarray,
#                               accuracy_in_t_deltaR_propto_deltap: np.ndarray,
#                               accuracy_in_t_deltaR_propto_Q: np.ndarray,
#                               accuracy_in_t_deltaR_propto_Power: np.ndarray,
#                               Colorscheme: "Color_Scheme", smooth: bool = True, window_size: int = 5):
#     """
#     Plots the accuracy in time for the Iris classification task using 4 materials.

#     input:
#     t_final        - int, final time step
#     t_for_accuracy - array of ints, times during simulation when accuracy was calculated
#     accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
#     dataset_shape  - shape of dataset used, for Iris it is [150, ?]
#     Colorscheme    - Object with a `.cmap` attribute defining the colormap
#     smooth         - boolean of whether to perform moving mean on test accuracy
#     window_size    - int, moving mean window

#     output:
#     plot of accuracy a.f.o time with confidence bounds as STD over ensemble
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     dataset_len = dataset_shape[0]
#     opacity = 0.25

#     material_keys = ['deltaR_deltap', 'deltaR_Q', 'deltaR_Power', 'R_deltap']
#     legend = [r'$\dot{R} \propto \Delta p$',
#               r'$\dot{R} \propto Q$',
#               r'$\dot{R} \propto \mathrm{Power}$',
#               r'$R \propto \Delta p$']
#     accuracy_data = [accuracy_in_t_deltaR_propto_deltap,
#                      accuracy_in_t_deltaR_propto_Q,
#                      accuracy_in_t_deltaR_propto_Power,
#                      accuracy_in_t_R_propto_deltap]

#     mean_accuracies = {}
#     std_accuracies = {}

#     for key, data in zip(material_keys, accuracy_data):
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#         if smooth:
#             mean = statistics.mov_ave(mean, window_size)
#             std = statistics.mov_ave(std, window_size)
#         mean_accuracies[key] = mean
#         std_accuracies[key] = std

#     if smooth:
#         t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracies['R_deltap'])]
#     else:
#         t_for_accuracy_smoothed = t_for_accuracy

#     # test accuracy for untrained network is 33%
#     for key in material_keys:
#         mean_accuracies[key][0] = 1 / 3

#     # Vertical lines to indicate dataset cycles
#     for t in range(t_final):
#         if t % dataset_len == 0:
#             plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1)

#     # Plotting
#     line_styles = ['-', '-', '--', '--']
#     for i, key in enumerate(material_keys):
#         plt.plot(t_for_accuracy_smoothed, mean_accuracies[key],
#                  color=Colorscheme.colors_lst[i],
#                  linestyle=line_styles[i], linewidth=3, alpha=1., marker=None)

#     for i, key in enumerate(material_keys):
#         plt.fill_between(t_for_accuracy_smoothed,
#                          mean_accuracies[key] - std_accuracies[key],
#                          mean_accuracies[key] + std_accuracies[key],
#                          color=Colorscheme.colors_lst[i], alpha=opacity)

#     for i in range(4):
#         plt.plot([], [], color=Colorscheme.colors_lst[i], label=legend[i])

#     # axes
#     plt.xlabel('$t$', fontsize=14)
#     plt.ylabel('Test accuracy', fontsize=14)
#     plt.ylim([0, 1])
#     plt.legend(loc='best')

#     set_thicker_spines(plt.gca(), linewidth=1.5)
#     plt.show()


# def plot_comparison_pseudo(R_pseudo: NDArray[np.float_], R_network: NDArray[np.float_],
#                            loss_pseudo: NDArray[np.float_], loss_network: NDArray[np.float_]) -> None:
#     """
#     plot comparison of performance of network to those of resistances calculated using
#     pseudo inverse method, as in the matlab file "Calculate_desired_resistances_2in3out_theoretical.m"
#     one plot with 2 subfigures of
#     1) resistances in time
#     2) loss in time
#     calculated by pseudo inverse (dashed) and network (solid)

#     inputs:
#     R_pseudo     - resistances calculated using pseudo inverse
#     R_network    - resistances of network in time
#     loss_pseudo  - loss in time using those found using pseudo inverse (resistances are constant in t)
#     loss_network - loss in time using the network (resistances change)
#     State   - class instance of the state variables of network
#     Variabs - class instance of the variables by the user

#     outputs:
#     1 matplotlib plot
#     """

#     # Setup
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     network_color = 'blue'
#     pseudo_color = 'violet'
#     legend2 = ['Network', 'Pseudo Inverse']

#     # Plot resistances in time (ax1)
#     network_lines = []
#     for i in range(R_network.shape[1]):
#         line = ax1.plot(R_network[:, i], color=network_color)
#         network_lines.append(line[0])  # Append the first line object from the plot

#     # Plot the pseudo inverse line (dashed and violet)
#     pseudo_line = ax1.plot(R_pseudo * np.ones([len(R_network), 1]), linestyle='--', color=pseudo_color)[0]

#     # Create a custom legend
#     # Only take one of the network lines since they all share the same color and appearance
#     ax1.legend([network_lines[0], pseudo_line], legend2, loc='best')
#     ax1.set_title(r'$R$')

#     # Plot loss in time (ax2)
#     ax2.plot(np.mean(np.mean(np.abs(loss_network), axis=1), axis=1), label='Network', color=network_color)
#     ax2.plot(np.mean(np.mean(np.abs(loss_pseudo), axis=1), axis=1), linestyle='--', label='Pseudo Inverse',
#              color=pseudo_color)
#     ax2.set_title('|Loss|')
#     ax2.set_xlabel('t')
#     ax2.set_yscale('log')
#     ax2.legend(legend2, loc='best')

#     plt.show()


# def plot_compare_R_type_loss(Network_1in2out: nx.DiGraph, Network_2in1out: nx.DiGraph,
#                              pos_lattice: dict,
#                              loss_1in2out_R_propto_deltap: NDArray[np.float_],
#                              loss_1in2out_deltaR_propto_deltap: NDArray[np.float_],
#                              loss_1in2out_propto_Q: NDArray[np.float_],
#                              loss_1in2out_propto_Power: NDArray[np.float_],
#                              loss_2in1out_R_propto_deltap: NDArray[np.float_],
#                              loss_2in1out_deltaR_propto_deltap: NDArray[np.float_],
#                              loss_2in1out_propto_Q: NDArray[np.float_],
#                              loss_2in1out_propto_Power: NDArray[np.float_]):
#     t = np.shape(loss_1in2out_propto_Power)[0]
#     range_vec = range(t)
#     range_vec = range(t)
#     t_vec = copy.copy(range_vec)
#     t_short = 100

#     legend = [r'$R \propto \Delta p$',
#               r'$\Delta R \propto \Delta p$',
#               r'$\Delta R \propto Q$',
#               r'$\Delta R \propto \mathrm{Power}$']

#     # Initialize broken axes for the top-left plot (ax1)
#     fig = plt.figure(figsize=(8, 8))
#     gs = fig.add_gridspec(2, 2)  # Define 2x2 grid layout

#     # Create broken axis in the top-left plot
#     baxtop = brokenaxes(xlims=((0, t_short), (t-40, t)), hspace=.05, subplot_spec=gs[0, 0])  # Constrain to top-left

#     # Plot data in the broken axis
#     baxtop.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_1in2out_R_propto_deltap), axis=1), axis=1)[:t_short],
#                 label=legend[0])
#     baxtop.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_1in2out_deltaR_propto_deltap), axis=1), axis=1)[:t_short],
#                 label=legend[1])
#     baxtop.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_1in2out_propto_Q), axis=1), axis=1)[:t_short],
#                 label=legend[2])
#     baxtop.plot(t_vec[:], np.mean(np.mean(np.abs(loss_1in2out_propto_Power), axis=1), axis=1)[:],
#                 label=legend[3])

#     # Extend the plot from t_short to t_full (the right side of the broken axis)
#     # bax.plot(t_vec[t_short:], np.mean(np.mean(np.abs(loss_1in2out_R_propto_deltap), axis=1), axis=1)[t_short:])
#     baxtop.set_ylabel(r'$\|\mathcal{L}\|$')

#     # Add legend
#     baxtop.legend(loc='best')

#     # Now create the rest of the figure manually
#     ax2 = fig.add_subplot(222)
#     nx.draw_networkx(Network_2in1out, pos=pos_lattice, edge_color='b', node_color='b', with_labels=True,
#                      font_color='white', font_size=14, ax=ax2)

#     # Create broken axis in the top-left plot
#     baxbot = brokenaxes(xlims=((0, t_short), (t-40, t)), hspace=.05, subplot_spec=gs[1, 0])  # Constrain to top-left

#     # Plot data in the broken axis
#     baxbot.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_2in1out_R_propto_deltap), axis=1), axis=1)[:t_short],
#                 label=legend[0])
#     baxbot.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_2in1out_deltaR_propto_deltap), axis=1), axis=1)[:t_short],
#                 label=legend[1])
#     baxbot.plot(t_vec[:t_short], np.mean(np.mean(np.abs(loss_2in1out_propto_Q), axis=1), axis=1)[:t_short],
#                 label=legend[2])
#     baxbot.plot(t_vec[:], np.mean(np.mean(np.abs(loss_2in1out_propto_Power), axis=1), axis=1)[:],
#                 label=legend[3])
#     baxbot.set_xlabel('t')
#     baxbot.set_ylabel(r'$\|\mathcal{L}\|$')
#     baxbot.legend(loc='best')

#     ax4 = fig.add_subplot(224)
#     nx.draw_networkx(Network_1in2out, pos=pos_lattice, edge_color='b', node_color='b', with_labels=True,
#                      font_color='white', font_size=14, ax=ax4)

#     plt.show()


# def plot_comparison_R_type(R_propto_deltap: NDArray[np.float_], deltaR_propto_deltap: NDArray[np.float_],
#                            deltaR_propto_Q: NDArray[np.float_], deltaR_propto_Power: NDArray[np.float_],
#                            loss_R_propto_deltap: NDArray[np.float_],
#                            loss_deltaR_propto_deltap: NDArray[np.float_],
#                            loss_propto_Q: NDArray[np.float_],
#                            loss_propto_Power: NDArray[np.float_]) -> None:
#     """
#     plot comparison of performance of network to those of resistances calculated using
#     pseudo inverse method, as in the matlab file "Calculate_desired_resistances_2in3out_theoretical.m"
#     one plot with 2 subfigures of
#     1) resistances in time
#     2) loss in time
#     calculated by pseudo inverse (dashed) and network (solid)

#     inputs:
#     R_pseudo     - resistances calculated using pseudo inverse
#     R_network    - resistances of network in time
#     loss_pseudo  - loss in time using those found using pseudo inverse (resistances are constant in t)
#     loss_network - loss in time using the network (resistances change)
#     State   - class instance of the state variables of network
#     Variabs - class instance of the variables by the user

#     outputs:
#     1 matplotlib plot
#     """

#     # setups
#     fig, axs = plt.subplots(2, 4, figsize=(12, 4))
#     (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8) = axs
#     R_color = 'blue'
#     legend1 = r'$R$'
#     legend2 = '|Loss|'

#     # Titles for the plots
#     titles = [
#         r'$R \propto \Delta p$',
#         r'$\Delta R \propto \Delta p$',
#         r'$R \propto Q$',
#         r'$\Delta R \propto \mathrm{Power}$'
#     ]

#     # Data for the plots
#     resistance_data = [R_propto_deltap, deltaR_propto_deltap, deltaR_propto_Q, deltaR_propto_Power]
#     loss_data = [loss_R_propto_deltap, loss_deltaR_propto_deltap, loss_propto_Q, loss_propto_Power]

#     # Manually set the y-axis sharing for the top row
#     for ax in [ax2, ax3, ax4]:
#         ax.sharey(ax1)  # Share y-axis with the first subplot (ax1)

#     # Plot resistance data (top row)
#     for ax, data, title in zip([ax1, ax2, ax3, ax4], resistance_data, titles):
#         ax.plot(data)
#         ax.set_title(title)
#         ax.legend([legend1], loc='best')

#     # Plot loss data (bottom row) with independent y-axes
#     for ax, data, title in zip([ax5, ax6, ax7, ax8], loss_data, titles):
#         ax.plot(np.mean(np.abs(data), axis=1), color=R_color)
#         ax.legend([legend2], loc='best')
#         ax.set_xlabel('t')
#         # ax.set_yscale('log')  # Logarithmic scale, auto-scaled to data

#     plt.show()
