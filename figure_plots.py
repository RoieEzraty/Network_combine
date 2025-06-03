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


import statistics

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State
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
                       Colorscheme: "Color_Scheme") -> None:
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
    node_colors_1in2out = [Colorscheme.colors_lst[1], Colorscheme.colors_lst[0], Colorscheme.colors_lst[0], 'black']
    # Draw arrows for ax4
    # draw_arrow(ax1, pos_lattice_both, 0, 1, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # in to output
    # draw_arrow(ax1, pos_lattice_both, 0, 2, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # in to output
    # draw_arrow(ax1, pos_lattice_both, 1, 3, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # out to ground
    # draw_arrow(ax1, pos_lattice_both, 2, 3, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # out to ground
    nx.draw_networkx(NET_1in2out, pos=pos_lattice_both, edge_color=Colorscheme.colors_lst[0],
                     node_color=node_colors_1in2out, with_labels=False, arrows=True, font_color='white',
                     font_size=14, width=2, node_size=400, ax=ax1)
    # Add custom labels
    nx.draw_networkx_labels(NET_1in2out, pos=pos_lattice_both, labels=label_dict_1in2out,
                            font_size=16, font_color='white', ax=ax1)
    ax1.set_title('Network structure')

    # "update" modality pressures
    ax2.plot(input_update_1in2out[1:T])
    ax2.plot(output_update_1in2out[1:T])
    ax2.set_title('"Update" modality pressure')
    ax2.legend(legend2_1in2out, loc='center right')

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
    node_colors_2in1out = [Colorscheme.colors_lst[1], Colorscheme.colors_lst[0], Colorscheme.colors_lst[1], 'black']
    # Draw arrows for ax5
    # draw_arrow(ax5, pos_lattice_both, 0, 2, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # in to output
    # draw_arrow(ax5, pos_lattice_both, 1, 2, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # in to output
    # draw_arrow(ax5, pos_lattice_both, 2, 3, color=Colorscheme.colors_lst[0], head_width=arrow_head_w)  # out to ground
    nx.draw_networkx(NET_2in1out, pos=pos_lattice_both, edge_color=Colorscheme.colors_lst[0],
                     node_color=node_colors_2in1out, with_labels=False, arrows=True, font_color='white',
                     font_size=14, width=2, node_size=400, ax=ax5)
    # Add custom labels
    nx.draw_networkx_labels(NET_1in2out, pos=pos_lattice_both, labels=label_dict_2in1out,
                            font_size=16, font_color='white', ax=ax5)

    # "update" modality pressures
    ax6.plot(input_update_2in1out[1:T])
    ax6.plot(output_update_2in1out[1:T])
    ax6.set_xlabel('t')
    ax6.legend(legend2_2in1out, loc='center right', bbox_to_anchor=(1, 0.4))

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

    plt.show()


def loss_afo_in_out(loss_mat_lin: np.ndarray, loss_mat_nonlin: np.ndarray, Colorscheme: "Color_Scheme") -> None:
    """
    Two-panel plot comparing linear and nonlinear update rules - ensemble mean of loss at end of training.

    Parameters:
    -----------
    loss_mat_lin : np.ndarray
        3D array [Nin, Nout, ensemble] for the linear system
    loss_mat_nonlin : np.ndarray
        3D array [Nin, Nout, ensemble] for the nonlinear system
    Colorscheme : Color_Scheme
        Object with a `.cmap` attribute defining the colormap

    outputs:
    matplotlib plot
    """
    v_max = 0.1
    plot_from = 0
    loss_mean_lin = np.mean(loss_mat_lin, axis=2)[plot_from:, plot_from:]
    loss_mean_nonlin = np.mean(loss_mat_nonlin, axis=2)[plot_from:, plot_from:]

    Nin = np.arange(plot_from+1, loss_mat_lin.shape[0]+1)  # array input dimension
    Nout = np.arange(plot_from+1, loss_mat_lin.shape[1]+1)  # array output dimension

    # instantiate figure and grid for positioning colorbal
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # cax = fig.add_subplot(gs[2])  # no need to specify colorbar here

    # linear update rule
    ax1.imshow(loss_mean_lin, cmap=Colorscheme.cmap, origin='lower',
               extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5],
               vmin=0, vmax=v_max)
    ax1.set_title(r'$\dot{R} \propto \Delta p$')
    ax1.set_xlabel('# Outputs')
    ax1.set_ylabel('# Inputs')
    ax1.set_xticks(Nin)
    ax1.set_yticks(Nout)
    set_thicker_spines(ax1, linewidth=1.5)

    # nonlinear update rule
    im2 = ax2.imshow(loss_mean_nonlin, cmap=Colorscheme.cmap, origin='lower',
                     extent=[min(Nin)-0.5, max(Nin)+0.5, min(Nout)-0.5, max(Nout)+0.5],
                     vmin=0, vmax=v_max)
    ax2.set_title(r'$\dot{R} \propto \left(\Delta p\right)^3$')
    ax2.set_xlabel('# Outputs')
    ax2.set_xticks(Nin)
    ax2.set_yticks(Nout)
    set_thicker_spines(ax2, linewidth=1.5)

    # colorbar axis positioned at [1.05, 0] with 0.08 width and 1.0 height
    cax = ax2.inset_axes((1.05, 0, 0.08, 1.0))
    # Add colorbar to the dedicated axis
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label(r'$\|\mathcal{L}\|$')

    plt.show()


def plot_accuracy_1_material(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                             dataset_shape: NDArray[np.int_], Colorscheme: "Color_Scheme", Iris_PNG_folder: str,
                             smooth: bool = True, window_size: int = 5) -> None:
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

    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(int(t_final)):
        if t % dataset_shape[0] == 0:
            plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1, alpha=0.3)

    # plot accuracy a.f.o time
    plt.plot(t_for_accuracy_smoothed, mean_accuracy, label='accuracy', color=Colorscheme.colors_lst[0], marker='.',
             linestyle='')

    # Plot confidence intervals using fill_between
    plt.fill_between(t_for_accuracy_smoothed, mean_accuracy - std,
                     mean_accuracy + std, color=Colorscheme.colors_lst[0], alpha=opacity)

    # axes
    plt.xlabel('$t$', fontsize=14)  # Set x-axis label with font size
    plt.xlim([-50, T*15])
    plt.ylabel('Test accuracy', fontsize=14)  # Set y-axis label with font size
    plt.ylim([0, 1])

    # Load your PNG image
    iris_img = mpimg.imread(Iris_PNG_folder)  # ← path to your image

    # Create the image box (tweak zoom as needed)
    imagebox = OffsetImage(iris_img, zoom=0.16)

    # Anchor it to lower right using axes fraction coordinates
    ab = AnnotationBbox(imagebox, (0.97, 0.08),  # x, y in axes coords
                        frameon=False,
                        xycoords='axes fraction',
                        box_alignment=(1, 0))  # align bottom-right corner of image to point

    # Add it to the current axes
    plt.gca().add_artist(ab)

    # Thicker spines
    set_thicker_spines(plt.gca(), linewidth=1.5)  # apply to the current Axes


def plot_final_accuracy_bar_chart(accuracy_in_t_R_propto_deltap: np.ndarray,
                                  accuracy_in_t_deltaR_propto_deltap_nonlin: np.ndarray,
                                  accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                                  accuracy_in_t_deltaR_propto_Q: np.ndarray,
                                  accuracy_in_t_deltaR_propto_Power: np.ndarray,
                                  Colorscheme: "Color_Scheme"):
    """
    Plots a bar chart of final average test accuracy over the last 46 time points.

    Each bar corresponds to a material type, with error bars showing std over 8 trials.
    """
    material_keys = ['deltaR_deltap', 'deltaR_deltap_nonlin', 'R_deltap', 'deltaR_Q', 'deltaR_Power']
    legend = [r'$\dot{R} \propto \Delta p$',
              r'$\dot{R} \propto \left(\Delta p\right)^3$',
              r'$R \propto \Delta p$',
              r'$\dot{R} \propto Q$',
              r'$\dot{R} \propto \mathrm{Power}$']
    accuracy_data = [accuracy_in_t_deltaR_propto_deltap,
                     accuracy_in_t_deltaR_propto_deltap_nonlin,
                     accuracy_in_t_R_propto_deltap,
                     accuracy_in_t_deltaR_propto_Q,
                     accuracy_in_t_deltaR_propto_Power]

    means = []
    stds = []

    for data in accuracy_data:
        final_46 = data[:, -46:]  # shape: (8, 46)
        mean_per_trial = np.mean(final_46, axis=1)  # shape: (8,)
        means.append(np.mean(mean_per_trial))       # scalar
        stds.append(np.std(mean_per_trial))         # scalar

    # Plotting
    x = np.arange(len(material_keys))
    colors = Colorscheme.colors_lst[:len(material_keys)]

    plt.figure(figsize=(6, 3))
    bars = plt.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
    plt.xticks(x, legend, rotation=20, fontsize=13)
    plt.ylabel("Final test accuracy", fontsize=14)
    plt.ylim([0.33, 1])
    plt.yticks(np.arange(0.3, 1.01, 0.1))  # Tick marks every 0.1
    plt.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.4)

    set_thicker_spines(plt.gca(), linewidth=1.5)
    plt.tight_layout()
    plt.show()


def plot_accuracy_5_materials(t_final: int, dataset_shape: np.ndarray, t_for_accuracy: np.ndarray,
                              accuracy_in_t_R_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_deltap_nonlin: np.ndarray,
                              accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_Q: np.ndarray,
                              accuracy_in_t_deltaR_propto_Power: np.ndarray,
                              Colorscheme: "Color_Scheme", smooth: bool = True, window_size: int = 5):
    """
    Plots the accuracy in time for the Iris classification task using 4 materials.

    input:
    t_final        - int, final time step
    t_for_accuracy - array of ints, times during simulation when accuracy was calculated
    accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
    dataset_shape  - shape of dataset used, for Iris it is [150, ?]
    Colorscheme    - Object with a `.cmap` attribute defining the colormap
    smooth         - boolean of whether to perform moving mean on test accuracy
    window_size    - int, moving mean window

    output:
    plot of accuracy a.f.o time with confidence bounds as STD over ensemble
    """
    import numpy as np
    import matplotlib.pyplot as plt

    dataset_len = dataset_shape[0]
    opacity = 0.25

    material_keys = ['deltaR_deltap', 'deltaR_deltap_nonlin', 'deltaR_Q', 'deltaR_Power', 'R_deltap']
    legend = [r'$\dot{R} \propto \Delta p$',
              r'$\dot{R} \propto \left(\Delta p\right)^3$',
              r'$\dot{R} \propto Q$',
              r'$\dot{R} \propto \mathrm{Power}$',
              r'$R \propto \Delta p$']
    accuracy_data = [accuracy_in_t_deltaR_propto_deltap,
                     accuracy_in_t_deltaR_propto_deltap_nonlin,
                     accuracy_in_t_deltaR_propto_Q,
                     accuracy_in_t_deltaR_propto_Power,
                     accuracy_in_t_R_propto_deltap]

    mean_accuracies = {}
    std_accuracies = {}

    for key, data in zip(material_keys, accuracy_data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if smooth:
            mean = statistics.mov_ave(mean, window_size)
            std = statistics.mov_ave(std, window_size)
        mean_accuracies[key] = mean
        std_accuracies[key] = std/2

    if smooth:
        t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracies['R_deltap'])]
    else:
        t_for_accuracy_smoothed = t_for_accuracy

    # test accuracy for untrained network is 33%
    for key in material_keys:
        mean_accuracies[key][0] = 1 / 3

    # Vertical lines to indicate dataset cycles
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1)

    # Plotting
    line_styles = ['-', '-', '-', '--', '--']
    for i, key in enumerate(material_keys):
        plt.plot(t_for_accuracy_smoothed, mean_accuracies[key],
                 color=Colorscheme.colors_lst[i],
                 linestyle=line_styles[i], linewidth=3, alpha=1., marker=None)

    for i, key in enumerate(material_keys):
        plt.fill_between(t_for_accuracy_smoothed,
                         mean_accuracies[key] - std_accuracies[key],
                         mean_accuracies[key] + std_accuracies[key],
                         color=Colorscheme.colors_lst[i], alpha=opacity)

    for i in range(4):
        plt.plot([], [], color=Colorscheme.colors_lst[i], label=legend[i])

    # axes
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Test accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(loc='best')

    set_thicker_spines(plt.gca(), linewidth=1.5)
    plt.show()


def plot_accuracy_4_materials(t_final: int, dataset_shape: np.ndarray, t_for_accuracy: np.ndarray,
                              accuracy_in_t_R_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_deltap: np.ndarray,
                              accuracy_in_t_deltaR_propto_Q: np.ndarray,
                              accuracy_in_t_deltaR_propto_Power: np.ndarray,
                              Colorscheme: "Color_Scheme", smooth: bool = True, window_size: int = 5):
    """
    Plots the accuracy in time for the Iris classification task using 4 materials.

    input:
    t_final        - int, final time step
    t_for_accuracy - array of ints, times during simulation when accuracy was calculated
    accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
    dataset_shape  - shape of dataset used, for Iris it is [150, ?]
    Colorscheme    - Object with a `.cmap` attribute defining the colormap
    smooth         - boolean of whether to perform moving mean on test accuracy
    window_size    - int, moving mean window

    output:
    plot of accuracy a.f.o time with confidence bounds as STD over ensemble
    """
    import numpy as np
    import matplotlib.pyplot as plt

    dataset_len = dataset_shape[0]
    opacity = 0.25

    material_keys = ['deltaR_deltap', 'deltaR_Q', 'deltaR_Power', 'R_deltap']
    legend = [r'$\dot{R} \propto \Delta p$',
              r'$\dot{R} \propto Q$',
              r'$\dot{R} \propto \mathrm{Power}$',
              r'$R \propto \Delta p$']
    accuracy_data = [accuracy_in_t_deltaR_propto_deltap,
                     accuracy_in_t_deltaR_propto_Q,
                     accuracy_in_t_deltaR_propto_Power,
                     accuracy_in_t_R_propto_deltap]

    mean_accuracies = {}
    std_accuracies = {}

    for key, data in zip(material_keys, accuracy_data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if smooth:
            mean = statistics.mov_ave(mean, window_size)
            std = statistics.mov_ave(std, window_size)
        mean_accuracies[key] = mean
        std_accuracies[key] = std

    if smooth:
        t_for_accuracy_smoothed = t_for_accuracy[:len(mean_accuracies['R_deltap'])]
    else:
        t_for_accuracy_smoothed = t_for_accuracy

    # test accuracy for untrained network is 33%
    for key in material_keys:
        mean_accuracies[key][0] = 1 / 3

    # Vertical lines to indicate dataset cycles
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color=Colorscheme.red, linestyle='--', linewidth=1)

    # Plotting
    line_styles = ['-', '-', '--', '--']
    for i, key in enumerate(material_keys):
        plt.plot(t_for_accuracy_smoothed, mean_accuracies[key],
                 color=Colorscheme.colors_lst[i],
                 linestyle=line_styles[i], linewidth=3, alpha=1., marker=None)

    for i, key in enumerate(material_keys):
        plt.fill_between(t_for_accuracy_smoothed,
                         mean_accuracies[key] - std_accuracies[key],
                         mean_accuracies[key] + std_accuracies[key],
                         color=Colorscheme.colors_lst[i], alpha=opacity)

    for i in range(4):
        plt.plot([], [], color=Colorscheme.colors_lst[i], label=legend[i])

    # axes
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Test accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(loc='best')

    set_thicker_spines(plt.gca(), linewidth=1.5)
    plt.show()


# Define a function to apply thicker spines globally
def set_thicker_spines(ax, linewidth=2):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


# # APPENDICES


def plot_comparison_Adaline(R_3in1out_Adaline: NDArray[np.float_],
                            R_3in1out_ourscheme: NDArray[np.float_],
                            loss_3in1out_Adaline: NDArray[np.float_],
                            loss_3in1out_ourscheme: NDArray[np.float_],
                            Colorscheme: "Color_Scheme", smooth: bool = True,
                            window_size: int = 10) -> None:

    # Set color cycle globally
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

    # Normalize R values so maximal will be 1
    R_3in1out_Adaline_norm = R_3in1out_Adaline[-1] / np.max(R_3in1out_Adaline[-1])
    R_3in1out_ourscheme_norm = R_3in1out_ourscheme[-1] / np.max(R_3in1out_ourscheme[-1])

    R_3in1out_Adaline_norm = np.concatenate([R_3in1out_Adaline_norm[:3], R_3in1out_Adaline_norm[-1:]])
    R_3in1out_ourscheme_norm = np.concatenate([R_3in1out_ourscheme[:3],  R_3in1out_ourscheme[-1:]])

    # only use run up to t=5600
    T = 6000

    # MAE Loss up to t=T
    loss_3in1out_Adaline_norm = np.mean(np.mean(np.abs(loss_3in1out_Adaline), axis=1), axis=1)[:T]
    loss_3in1out_ourscheme_norm = np.mean(np.mean(np.abs(loss_3in1out_ourscheme), axis=1), axis=1)[:T]

    if smooth:
        loss_3in1out_Adaline_norm = statistics.mov_ave(loss_3in1out_Adaline_norm, window_size)
        loss_3in1out_ourscheme_norm = statistics.mov_ave(loss_3in1out_ourscheme_norm, window_size)

    # weird setup for bars
    x = np.arange(len(R_3in1out_ourscheme_norm))
    bar_width = 0.35

    # instantiate figure and grid for positioning colorbal
    fig = plt.figure(figsize=(7, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.45)

    # R bar plot
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x - bar_width / 2, R_3in1out_Adaline_norm,
            width=bar_width, label='Adaline-like', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax0.bar(x + bar_width / 2, R_3in1out_Adaline_norm,
            width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax0.set_xticks([0, 1, 2, 3])
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_ylabel('$R$')
    # ax0.legend()

    # Loss plot
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(loss_3in1out_Adaline_norm, label='Adaline-like')
    ax1.plot(loss_3in1out_ourscheme_norm, label='this work')
    ax1.set_ylabel(r'$\|\mathcal{L}\|$')
    ax1.set_xlabel(r'$t$')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3, 1)
    ax1.legend()

    # Thicker spines
    for ax in [ax0, ax1]:
        set_thicker_spines(ax)

    plt.tight_layout()
    plt.show()


def plot_comparison_GD(R_mine_1in2out: NDArray[np.float_], R_GD_1in2out: NDArray[np.float_],
                       R_mine_2in1out: NDArray[np.float_], R_GD_2in1out: NDArray[np.float_],
                       cosine_sim_1in2out: NDArray[np.float_], cosine_sim_2in1out: NDArray[np.float_],
                       Colorscheme: "Color_Scheme") -> None:

    """
    Two rows plot with 3 subfigures each
    1) Bar plot of resistances at end of training using gradient descent (GD) and proposed scheme
    2) loss a.f.o t using GD and proposed scheme
    3) cosine similarity between change in conductivities using GD and my scheme

    inputs:
    too many

    outputs:
    matplotlib plot
    """
    # Set color cycle globally
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', Colorscheme.colors_lst)

    # Normalize R values so maximal will be 1
    R_mine_1in2out_norm = R_mine_1in2out[-1] / np.max(R_mine_1in2out[-1])
    R_GD_1in2out_norm = R_GD_1in2out[-1] / np.max(R_GD_1in2out[-1])
    R_mine_2in1out_norm = R_mine_2in1out[-1] / np.max(R_mine_2in1out[-1])
    R_GD_2in1out_norm = R_GD_2in1out[-1] / np.max(R_GD_2in1out[-1])

    # omit input to ground
    R_mine_1in2out_norm = np.concatenate([R_mine_1in2out_norm[:2], R_mine_1in2out_norm[-2:]])
    R_GD_1in2out_norm = np.concatenate([R_GD_1in2out_norm[:2], R_GD_1in2out_norm[-2:]])
    R_mine_2in1out_norm = np.concatenate([R_mine_2in1out_norm[:2], R_mine_2in1out_norm[-1:]])
    R_GD_2in1out_norm = np.concatenate([R_GD_2in1out_norm[:2], R_GD_2in1out_norm[-1:]])

    # weird setup for bars
    x_1in2out = np.arange(len(R_GD_1in2out_norm))
    x_2in1out = np.arange(len(R_GD_2in1out_norm))
    bar_width = 0.35

    # Grid: 2 rows, 3 columns
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.2], height_ratios=[1, 1])

    # ---- Row 1 - 1 input 2 outputs ----

    # R bar plot
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x_1in2out - bar_width / 2, R_GD_1in2out_norm,
            width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax0.bar(x_1in2out + bar_width / 2, R_mine_1in2out_norm,
            width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_xticks([0, 1, 2, 3])
    ax0.set_title('$R$')
    ax0.legend()

    # Cosine similarity
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(cosine_sim_1in2out)
    ax1.plot(np.zeros([len(cosine_sim_1in2out)]), '--k')  # dotted line at cosine=0
    ax1.set_yticks([-1, 0, 1])
    ax1.set_title(r'$\cos\left(\dot{\vec{k}},\dot{\vec{k}}_{GD}\right)$')
    ax1.set_ylim(-1, 1)

    # ---- Row 1 - 2 inputs 1 output ----

    # R bar plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(x_2in1out - bar_width / 2, R_GD_2in1out_norm,
            width=bar_width, label='GD', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax2.bar(x_2in1out + bar_width / 2, R_mine_2in1out_norm,
            width=bar_width, label='this work', alpha=0.8, edgecolor='k', linewidth=1.6)
    ax2.set_xticks([0, 1, 2,])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_xlabel('edge #')
    ax2.legend()

    # Cosine similarity
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(cosine_sim_2in1out)
    ax3.plot(np.zeros([len(cosine_sim_2in1out)]), '--k')  # dotted line at cosine=0
    ax3.set_yticks([-1, 0, 1])
    ax3.set_xlabel('$t$')
    ax3.set_ylim(-1, 1)

    # Thicker spines
    for ax in [ax0, ax1, ax2, ax3]:
        set_thicker_spines(ax)

    plt.tight_layout()
    plt.show()


# # Auxiliary functions


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
