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
# functions for plots
# ================================


def plot_importants(State: "Network_State", Variabs: "User_Variables", desired: List[NDArray[np.float_]],
                    M: Optional[NDArray[np.int_]] = None, include_network: Optional[bool] = False,
                    NET: Optional[nx.DiGraph] = None) -> None:
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
    if Variabs.Nin == 1 and Variabs.Nout == 1:  # 1by1, simplest
        if M is not None:
            A: float = M[0]
            R_theor: NDArray[np.float_] = np.array([(1-A)/A])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
    elif Variabs.Nin == 1 and Variabs.Nout == 2:  # Allostery
        if M is not None:
            A = M[0]  # A = x_hat/p_in
            B: float = M[1]  # B = y_hat/p_in
            Rl_subs: float = 1.0
            R_theor = State.input_drawn_in_t[0]*np.array([(1-A)/(A*(1+1/Rl_subs)-B/Rl_subs),
                                                          (1-B)/(B*(1+1/Rl_subs)-A/Rl_subs)])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        # legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
    elif Variabs.Nin == 2 and Variabs.Nout == 1:  # Regression
        if M is not None:
            A = M[0, 0]
            B = M[0, 1]
            Rl_subs = 1
            R_theor = np.array([Rl_subs*(1-A-B)/A, Rl_subs*(1-A-B)/B])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        # legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_1\,\mathrm{theoretical}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
    elif Variabs.Nin == 2 and Variabs.Nout == 3:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$z\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$',
                   r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$', r'$R_6$']
    elif Variabs.Nin == 2 and Variabs.Nout == 2:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        if Variabs.access_interNodes:
            legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$',
                       r'$\mathrm{inter1\,dual}$', r'$\mathrm{inter2\,dual}$']
        else:
            legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$']
    elif Variabs.Nin == 3 and Variabs.Nout == 3:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$z\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$',
                   r'$p_2\,\mathrm{dual}$', r'$p_3\,\mathrm{dual}$']
        legend3 = []
    elif Variabs.task_type == 'Iris_classification':
        legend1 = [r'$\mathrm{Setosa}$', r'$\mathrm{Verisicolor}$', r'$\mathrm{Virginica}$']
        legend2 = [r'$\mathrm{Setosa\,dual}$', r'$\mathrm{Verisicolor\,dual}$',
                   r'$\mathrm{Virginica\,dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$',
                   r'$p_3\,\mathrm{dual}$', r'$p_4\,\mathrm{dual}$']
        legend3 = []
    else:
        legend1 = []
        legend2 = []
        legend3 = []
    legend4 = ['|loss|']
    if include_network:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 3))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.2))
    if Variabs.task_type != 'Iris_classification':
        ax1.plot(np.linspace(0, State.t, np.shape(State.output_in_t)[0]).T,
                 np.asarray(State.output_in_t)/np.asarray(State.desired_in_t)-1)
    else:
        ax1.plot(np.linspace(0, State.t, np.shape(State.output_in_t)[0]).T,
                 np.asarray(State.output_in_t))
    ax1.set_title('output in time')
    ax1.set_xlabel('t')
    if legend1:
        ax1.legend(legend1)
    ax2.plot(State.output_dual_in_t[1:])
    ax2.plot(State.input_dual_in_t[1:])
    if Variabs.access_interNodes:
        ax2.plot(State.inter_dual_in_t[1:])
    ax2.set_title('dual and p in time')
    ax2.set_xlabel('t')
    # ax2.set_ylim([-0.2,0.2])
    if legend2:
        ax2.legend(legend2)
    ax3.plot(State.R_in_t[1:])
    # if 'R_theor' in locals():  # if theoretical values were calculated, plot them
    #     print('R theoretical', R_theor)
    #     ax3.plot(np.outer(R_theor, np.ones(State.t)).T, '--')
    ax3.set_title('R in time')
    ax3.set_xlabel('t')
    if legend3:
        ax3.legend(legend3)
    for t in range(State.t):
        if t % len(Variabs.dataset) == 0 and t != 0:
            ax4.axvline(x=t, color='red', linestyle='--', linewidth=1)
    ax4.plot(np.mean(np.mean(np.abs(State.loss_in_t[1:]), axis=1), axis=1))
    ax4.set_xlabel('t')
    ax4.set_yscale('log')
    if legend4:
        ax4.legend(legend4)
    # fig.suptitle(f'alpha={Variabs.alpha_vec}')
    if include_network:
        if NET is not None:
            nx.draw_networkx(NET.NET, pos=NET.pos_lattice, edge_color='b', node_color='b', with_labels=True, ax=ax5)
        else:
            print('no NET assigned in input')
    plt.show()


def plotNetStructure(NET: nx.DiGraph, plot: bool = False, node_labels: bool = False) -> Dict[Any, Tuple[float, float]]:
    """
    Plots the structure (nodes and edges) of networkx NET

    input:
    NET         - networkx net of nodes and edges
    plot        - bool, whether to plot or not
    node_labels - boolean, show node number in plot or not

    output:
    pos_lattice - dict of positions of nodes from NET.nodes
    if plot=='yes' also show matplotlib plot of network structure
    """
    pos_lattice: Dict[Any, Tuple[float, float]] = nx.spring_layout(NET, k=1.0, iterations=20)
    if plot:
        nx.draw_networkx(NET, pos_lattice, edge_color='b', node_color='b', with_labels=node_labels)
        plt.show()
    print('NET is ready')
    return pos_lattice


def plot_accuracy(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                  dataset_len: np.int_) -> None:
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
    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1)

    # plot accuracy a.f.o time
    plt.plot(t_for_accuracy, accuracy_in_t, label='accuracy')

    # axes
    plt.xlabel('t', fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Accuracy', fontsize=14)  # Set y-axis label with font size
    plt.title('Accuracy Over Time', fontsize=16)  # Set title with font size


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
    A_2in1out = M[0, 0]
    B_2in1out = M[0, 1]
    R_theor_2in1out = np.array([(1-A_2in1out-B_2in1out)/A_2in1out, (1-A_2in1out-B_2in1out)/B_2in1out])
    legend1_2in1out = [r'$\frac{x}{x\,\mathrm{desired}}$']
    legend2_2in1out = [r'$x\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
    legend3_2in1out = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_1\,\mathrm{theoretical}$']

    # sizes for both
    legend4 = ['|loss|']

    # figure
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(12, 6.4))

    # plot 1 input 2 output
    ax1.plot(np.linspace(0, t, np.shape(output_1in2out)[0]).T, np.asarray(output_2in1out))
    ax1.set_title('output in time')
    ax1.legend(legend1_1in2out)
    ax2.plot(input_dual_1in2out)
    ax2.plot(output_dual_1in2out)
    ax2.set_title('dual and p in time')
    ax2.legend(legend2_1in2out)
    ax3.plot(R_1in2out)
    ax3.plot(np.outer(R_theor_1in2out, np.ones(t)).T, '--')
    ax3.set_title('R in time')
    ax3.legend(legend3_1in2out)
    ax4.plot(np.mean(np.mean(np.abs(loss_1in2out), axis=1), axis=1))
    ax4.set_yscale('log')
    ax4.legend(legend4)
    nx.draw_networkx(NET_1in2out, pos=pos_lattice_1in2out, edge_color='b', node_color='b', with_labels=True, ax=ax5)

    # plot 2 input 1 output
    ax6.plot(np.linspace(0, t, np.shape(output_2in1out)[0]).T, np.asarray(output_1in2out))
    ax6.set_title('output in time')
    ax6.set_xlabel('t')
    ax6.legend(legend1_2in1out)
    ax7.plot(input_dual_2in1out)
    ax7.plot(output_dual_2in1out)
    ax7.set_title('dual and p in time')
    ax7.set_xlabel('t')
    ax7.legend(legend2_2in1out)
    ax8.plot(R_2in1out)
    ax8.plot(np.outer(R_theor_2in1out, np.ones(t)).T, '--')
    ax8.set_title('R in time')
    ax8.set_xlabel('t')
    ax8.legend(legend3_2in1out)
    ax9.plot(np.mean(np.mean(np.abs(loss_2in1out), axis=1), axis=1))
    ax9.set_xlabel('t')
    ax9.set_yscale('log')
    ax9.legend(legend4)
    nx.draw_networkx(NET_2in1out, pos=pos_lattice_2in1out, edge_color='b', node_color='b', with_labels=True, ax=ax10)

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
    ax2.plot(np.mean(np.mean(np.abs(loss_pseudo), axis=1), axis=1), linestyle='--', label='Pseudo Inverse', color=pseudo_color)
    ax2.set_title('|Loss|')
    ax2.set_xlabel('t')
    ax2.set_yscale('log')
    ax2.legend(legend2, loc='best')

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
        ax.set_yscale('log')  # Logarithmic scale, auto-scaled to data

    plt.show()

# def PlotNetwork(p: np.ndarray, u: np.ndarray, K: np.ndarray, BigClass: "Big_Class", mode: str = 'u_p',
#                 nodes: bool = True, edges: bool = True, savefig: bool = False) -> None:
#     """
#     Plots the flow network structure alongside hydrostatic pressure, flows and conductivities
#     pressure denoted by colors from purple (high) to cyan (low)
#     flow velocity denoted by arrow direction and thickness
#     conductivity denoted by arrow color - black (low) and blue (high)

#     input:
#     p - 1D np.array sized [NNodes], hydrostatic pressure at nodes
#     u - 1D np.array sized [NEdges], flow velocities at edges (positive is into cell center)
#     K - 1D np.array sized [NEdges], flow conductivities for every edge
#     BigClass - Class instance with all inside class instances User_variables, Network_State, etc
#     mode: str, what to plot: 'u_p' = plot velocity and pressure
#                              'R' = plot resistances and pressure
#     nodes: bool, plot the nodes or not
#     edges: bool, plot the edges (as velocity or resistances) or not
#     savefig: bool, save figure into PNG file or not

#     output:
#     matplotlib plot of network structure, possibly with saved file
#     """
#     NN = BigClass.Strctr.NN
#     NE = BigClass.Strctr.NE
#     EIEJ_plots = BigClass.Strctr.EIEJ_plots
#     NET = BigClass.NET.NET
#     pos_lattice = BigClass.NET.pos_lattice

#     # Preliminaries for the plot
#     node_sizes = 4*24
#     if mode == 'u_p':
#         u_rescale_factor = 5
#     elif mode == 'R':
#         R_rescale_factor = 5

#     # p values at nodes - the same in EIEJ and in networkx's NET
#     val_map = {i: p[i][0] for i in range(NN)}
#     values = [val_map.get(node, 0.25) for node in NET.nodes()]

#     # velocity and conductivity values at edges - not the same in EIEJ and in networkx's NET
#     NETEdges = list(NET.edges)

#     # rearrange u and K for network
#     # since NET.edges and EIEJ_plots are not the same, thanks networkx you idiot
#     u_NET = zeros(NE)  # velocity field arranged as edges in NET.edges and not EIEJ_plots
#     K_NET = zeros(NE)  # conductivity values at edges in NET.edges and not EIEJ_plots
#     for i in range(NE):
#         K_NET[i] = K[EIEJ_plots.index(NETEdges[i])]
#         u_NET[i] = u[EIEJ_plots.index(NETEdges[i])]

#     # Need to create a layout when doing
#     # separate calls to draw nodes and edges
#     if nodes:
#         nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('cool'),
#                                node_color=values, node_size=node_sizes)

#     if edges:
#         if mode == 'u_p':
#             positive_u_NET_inds = np.where(u_NET > 10**-10)[0]  # indices of edges with positive flow vel
#             negative_u_NET_inds = np.where(u_NET < -10**-10)[0]  # indices of edges with negative flow vel

#             r_edges_positive = [NETEdges[i] for i in positive_u_NET_inds]  # edges with low conductivity, positive flow
#             r_edges_negative = [NETEdges[i] for i in negative_u_NET_inds]  # edges with low conductivity, negative flow

#             # save arrow sizes
#             rescaled_u_NET = abs(u_NET)*u_rescale_factor/max(abs(u_NET))

#             edgewidths_k_positive = rescaled_u_NET[positive_u_NET_inds]
#             edgewidths_k_negative = rescaled_u_NET[negative_u_NET_inds]
#             # draw with right arrow widths ("width") and directions ("arrowstyle")
#             nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='k', arrows=True,
#                                    width=edgewidths_k_positive, arrowstyle='->')
#             nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='k', arrows=True,
#                                    width=edgewidths_k_negative, arrowstyle='<-')
#         elif mode == 'R_p':
#             nx.draw_networkx_edges(NET, pos_lattice, edgelist=NETEdges, edge_color='k', arrows=False, width=K_NET)

#     if savefig:
#         # prelims for figures
#         comp_path = "C:\\Users\\SMR_Admin\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
#         # comp_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
#         datenow = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
#         plt.savefig(comp_path + 'network_' + str(datenow) + '.png', bbox_s='tight')
#     plt.show()
