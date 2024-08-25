from __future__ import annotations
import numpy as np
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Any
from numpy import array, zeros
from datetime import datetime
from typing import TYPE_CHECKING
from numpy.typing import NDArray

from typing import Tuple, List, Union

import statistics

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State
    from Big_Class import Big_Class


# ================================
# functions for plots
# ================================


def plot_importants(State: "Network_State", Variabs: "User_Variables", desired: List[NDArray[np.float_]],
                    M: NDArray[np.int_]) -> None:
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
        A: float = M[0]
        R_theor: NDArray[np.float_] = np.array([(1-A)/A])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
    elif Variabs.Nin == 1 and Variabs.Nout == 2:  # Allostery
        A = M[0]  # A = x_hat/p_in
        B: float = M[1]  # B = y_hat/p_in
        Rl_subs: float = 1.0
        R_theor = State.input_drawn_in_t[0]*np.array([(1-A)/(A*(1+1/Rl_subs)-B/Rl_subs),
                                                      (1-B)/(B*(1+1/Rl_subs)-A/Rl_subs)])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
    elif Variabs.Nin == 2 and Variabs.Nout == 1:  # Regression
        A = M[0, 0]
        B = M[0, 1]
        Rl_subs = 1
        R_theor = np.array([Rl_subs*(1-A-B)/A, Rl_subs*(1-A-B)/B])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_1\,\mathrm{theoretical}$']
    elif Variabs.Nin == 2 and Variabs.Nout == 3:
        A = M[0, 0]
        B = M[0, 1]
        C: float = M[1, 0]
        D: float = M[1, 1]
        E: float = M[2, 0]
        F: float = M[2, 1]
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$z\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$',
                   r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$', r'$R_6$']
        R_theor = np.ndarray([])  # I didn't calculate it for this task
    elif Variabs.Nin == 2 and Variabs.Nout == 2:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$']
        R_theor = np.ndarray([])  # I didn't calculate it for this task
    legend4 = ['|loss|']
    print('R theoretical', R_theor)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.2))
    ax1.plot(np.linspace(0, State.t, 2*State.t-1).T,
             np.asarray(State.output_in_t[1:])/np.asarray(State.desired_in_t[1:])-1)

    ax1.set_title('output in time')
    ax1.set_xlabel('t')
    ax1.legend(legend1)
    ax2.plot(State.output_dual_in_t[1:])
    ax2.plot(State.input_dual_in_t[1:])
    ax2.set_title('dual and p in time')
    ax2.set_xlabel('t')
    # ax2.set_ylim([-0.2,0.2])
    ax2.legend(legend2)
    ax3.plot(State.R_in_t[1:])
    ax3.plot(np.outer(R_theor, np.ones(State.t)).T, '--')
    ax3.set_title('R in time')
    ax3.set_xlabel('t')
    ax3.legend(legend3)
    ax4.plot(np.mean(np.mean(np.abs(State.loss_in_t[1:]), axis=1), axis=1))
    ax4.set_xlabel('t')
    ax4.set_yscale('log')
    ax4.legend(legend4)
    fig.suptitle(f'alpha={Variabs.alpha_vec}')
    plt.show()


def plotNetStructure(NET: nx.DiGraph, plot=False, node_labels=False) -> Dict[Any, Tuple[float, float]]:
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


def PlotNetwork(p: np.ndarray, u: np.ndarray, K: np.ndarray, BigClass: "Big_Class", mode: str = 'u_p',
                nodes: bool = True, edges: bool = True, savefig: bool = False) -> None:
    """
    Plots the flow network structure alongside hydrostatic pressure, flows and conductivities
    pressure denoted by colors from purple (high) to cyan (low)
    flow velocity denoted by arrow direction and thickness
    conductivity denoted by arrow color - black (low) and blue (high)

    input:
    p - 1D np.array sized [NNodes], hydrostatic pressure at nodes
    u - 1D np.array sized [NEdges], flow velocities at edges (positive is into cell center)
    K - 1D np.array sized [NEdges], flow conductivities for every edge
    BigClass - Class instance with all inside class instances User_variables, Network_State, etc
    mode: str, what to plot: 'u_p' = plot velocity and pressure
                             'R' = plot resistances and pressure
    nodes: bool, plot the nodes or not
    edges: bool, plot the edges (as velocity or resistances) or not
    savefig: bool, save figure into PNG file or not

    output:
    matplotlib plot of network structure, possibly with saved file
    """
    NN = BigClass.Strctr.NN
    NE = BigClass.Strctr.NE
    EIEJ_plots = BigClass.Strctr.EIEJ_plots
    NET = BigClass.NET.NET
    pos_lattice = BigClass.NET.pos_lattice

    # Preliminaries for the plot
    node_sizes = 4*24
    if mode == 'u_p':
        u_rescale_factor = 5
    elif mode == 'R':
        R_rescale_factor = 5

    # p values at nodes - the same in EIEJ and in networkx's NET
    val_map = {i: p[i][0] for i in range(NN)}
    values = [val_map.get(node, 0.25) for node in NET.nodes()]

    # velocity and conductivity values at edges - not the same in EIEJ and in networkx's NET
    NETEdges = list(NET.edges)

    # rearrange u and K for network
    # since NET.edges and EIEJ_plots are not the same, thanks networkx you idiot
    u_NET = zeros(NE)  # velocity field arranged as edges in NET.edges and not EIEJ_plots
    K_NET = zeros(NE)  # conductivity values at edges in NET.edges and not EIEJ_plots
    for i in range(NE):
        K_NET[i] = K[EIEJ_plots.index(NETEdges[i])]
        u_NET[i] = u[EIEJ_plots.index(NETEdges[i])]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    if nodes:
        nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('cool'),
                               node_color=values, node_size=node_sizes)

    if edges:
        if mode == 'u_p':
            positive_u_NET_inds = np.where(u_NET > 10**-10)[0]  # indices of edges with positive flow vel
            negative_u_NET_inds = np.where(u_NET < -10**-10)[0]  # indices of edges with negative flow vel

            r_edges_positive = [NETEdges[i] for i in positive_u_NET_inds]  # edges with low conductivity, positive flow
            r_edges_negative = [NETEdges[i] for i in negative_u_NET_inds]  # edges with low conductivity, negative flow

            # save arrow sizes
            rescaled_u_NET = abs(u_NET)*u_rescale_factor/max(abs(u_NET))

            edgewidths_k_positive = rescaled_u_NET[positive_u_NET_inds]
            edgewidths_k_negative = rescaled_u_NET[negative_u_NET_inds]
            # draw with right arrow widths ("width") and directions ("arrowstyle")
            nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='k', arrows=True,
                                   width=edgewidths_k_positive, arrowstyle='->')
            nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='k', arrows=True,
                                   width=edgewidths_k_negative, arrowstyle='<-')
        elif mode == 'R_p':
            nx.draw_networkx_edges(NET, pos_lattice, edgelist=NETEdges, edge_color='k', arrows=False, width=K_NET)

    if savefig:
        # prelims for figures
        comp_path = "C:\\Users\\SMR_Admin\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
        # comp_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
        datenow = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        plt.savefig(comp_path + 'network_' + str(datenow) + '.png', bbox_s='tight')
    plt.show()
