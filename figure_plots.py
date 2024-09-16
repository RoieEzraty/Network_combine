from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Any
from typing import TYPE_CHECKING
from numpy.typing import NDArray

from typing import Tuple, List, Union, Optional

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State


# ================================
# functions for paper figure plots
# ================================


def plot_performance_2(M: NDArray[np.float_], t: np.int_,
                       output_1in2out: NDArray[np.float_], output_2in1out: NDArray[np.float_],
                       input_dual_1in2out: NDArray[np.float_], input_dual_2in1out: NDArray[np.float_],
                       output_dual_1in2out: NDArray[np.float_], output_dual_2in1out: NDArray[np.float_],
                       R_1in2out: NDArray[np.float_], R_2in1out: NDArray[np.float_],
                       loss_1in2out: NDArray[np.float_], loss_2in1out: NDArray[np.float_],
                       NET_1in2out: nx.DiGraph, NET_2in1out: nx.DiGraph,
                       pos_lattice_1in2out: dict, pos_lattice_2in1out: dict,) -> None:
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

    # sizes for 1 input 2 output
    A_1in2out: float = M[0]  # A = x_hat/p_in
    B_1in2out: float = M[1]  # B = y_hat/p_in
    R_theor_1in2out = np.array([(1-A_1in2out)/(A_1in2out*(1+1)-B_1in2out),
                                (1-B_1in2out)/(B_1in2out*(1+1)-A_1in2out)])
    legend1_1in2out = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
    legend2_1in2out = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
    legend3_1in2out = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']

    # sizes for 2 input 1 output
    A_2in1out = M[0]
    B_2in1out = M[1]
    R_theor_2in1out = np.array([(1-A_2in1out-B_2in1out)/A_2in1out, (1-A_2in1out-B_2in1out)/B_2in1out])
    legend1_2in1out = [r'$\frac{x}{x\,\mathrm{desired}}$']
    legend2_2in1out = [r'$x\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
    legend3_2in1out = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_1\,\mathrm{theoretical}$']

    # sizes for both
    legend4 = ['|loss|']
    pos_lattice_both = pos_lattice_2in1out

    # figure
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(17, 6))

    # plot 1 input 2 output
    ax1.plot(np.linspace(0, t, np.shape(output_1in2out)[0]).T, np.asarray(output_1in2out))
    ax1.set_title('output in time')
    ax1.legend(legend1_1in2out)
    ax2.plot(input_dual_1in2out)
    ax2.plot(output_dual_1in2out)
    ax2.set_title('dual and p in time')
    ax2.legend(legend2_1in2out)
    ax3.plot(R_1in2out)
    # ax3.plot(np.outer(R_theor_1in2out, np.ones(t)).T, '--')
    ax3.set_title('R in time')
    # ax3.legend(legend3_1in2out)
    ax4.plot(np.mean(np.mean(np.abs(loss_1in2out), axis=1), axis=1), color='blue')
    ax4.set_yscale('log')
    ax4.legend(legend4)
    nx.draw_networkx(NET_1in2out, pos=pos_lattice_both, edge_color='b', node_color='b', with_labels=True,
                     font_color='white', font_size=14, ax=ax5)

    # plot 2 input 1 output
    ax6.plot(np.linspace(0, t, np.shape(output_2in1out)[0]).T, np.asarray(output_2in1out))
    ax6.set_xlabel('t')
    ax6.legend(legend1_2in1out)
    ax7.plot(input_dual_2in1out)
    ax7.plot(output_dual_2in1out)
    ax7.set_xlabel('t')
    ax7.legend(legend2_2in1out)
    ax8.plot(R_2in1out)
    # ax8.plot(np.outer(R_theor_2in1out, np.ones(t)).T, '--')
    ax8.set_xlabel('t')
    # ax8.legend(legend3_2in1out)
    ax9.plot(np.mean(np.mean(np.abs(loss_2in1out), axis=1), axis=1), color='blue')
    ax9.set_xlabel('t')
    ax9.set_yscale('log')
    ax9.legend(legend4)
    nx.draw_networkx(NET_2in1out, pos=pos_lattice_both, edge_color='b', node_color='b', with_labels=True,
                     font_color='white', font_size=14, ax=ax10)

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


def plot_compare_R_type_loss(t: np.int_, Network_1in2out: nx.DiGraph, Network_2in1out: nx.DiGraph,
                             pos_lattice: dict,
                             loss_1in2out_R_propto_deltap: NDArray[np.float_],
                             loss_1in2out_deltaR_propto_deltap: NDArray[np.float_],
                             loss_1in2out_propto_Q: NDArray[np.float_],
                             loss_1in2out_propto_Power: NDArray[np.float_],
                             loss_2in1out_R_propto_deltap: NDArray[np.float_],
                             loss_2in1out_deltaR_propto_deltap: NDArray[np.float_],
                             loss_2in1out_propto_Q: NDArray[np.float_],
                             loss_2in1out_propto_Power: NDArray[np.float_]):

    t_vec = range(t)

    # figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    legend = [r'$R \propto \Delta p$',
              r'$\Delta R \propto \Delta p$',
              r'$\Delta R \propto Q$',
              r'$\Delta R \propto \mathrm{Power}$']

    # Plot for ax1
    ax1.plot(t_vec, np.mean(np.mean(np.abs(loss_1in2out_R_propto_deltap), axis=1), axis=1))
    ax1.plot(t_vec, np.mean(np.mean(np.abs(loss_1in2out_deltaR_propto_deltap), axis=1), axis=1))
    ax1.plot(t_vec, np.mean(np.mean(np.abs(loss_1in2out_propto_Q), axis=1), axis=1))
    ax1.plot(t_vec, np.mean(np.mean(np.abs(loss_1in2out_propto_Power), axis=1), axis=1))
    ax1.set_yscale('log')  # Logarithmic scale, auto-scaled to data
    ax1.set_ylabel(r'$\|\mathcal{L}\|$')
    ax1.legend(legend, loc='best')  # Corrected the legend call

    # Network plot for ax2
    nx.draw_networkx(Network_2in1out, pos=pos_lattice, edge_color='b', node_color='b', with_labels=True,
                     font_color='white', font_size=14, ax=ax2)

    # Plot for ax3
    ax3.plot(t_vec, np.mean(np.mean(np.abs(loss_2in1out_R_propto_deltap), axis=1), axis=1))
    ax3.plot(t_vec, np.mean(np.mean(np.abs(loss_2in1out_deltaR_propto_deltap), axis=1), axis=1))
    ax3.plot(t_vec, np.mean(np.mean(np.abs(loss_2in1out_propto_Q), axis=1), axis=1))
    ax3.plot(t_vec, np.mean(np.mean(np.abs(loss_2in1out_propto_Power), axis=1), axis=1))
    ax3.set_yscale('log')  # Logarithmic scale, auto-scaled to data
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\|\mathcal{L}\|$')
    ax3.legend(legend, loc='best')  # Legend here is correct as well

    # Network plot for ax4
    nx.draw_networkx(Network_1in2out, pos=pos_lattice, edge_color='b', node_color='b', with_labels=True,
                     font_color='white', font_size=14, ax=ax4)


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
        ax.set_yscale('log')  # Logarithmic scale, auto-scaled to data

    plt.show()
