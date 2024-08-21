from __future__ import annotations
import numpy as np
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Any
from numpy import array as array
from numpy import zeros as zeros
from typing import TYPE_CHECKING

import statistics

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State


############# functions for plots #############


def plot_importants(state: "Network_State", variabs: "User_Variables", desired: np.ndarray, A: int=1, B: int=1) -> None:
    if variabs.task_type == 'Allostery':
        A = desired[0]/state.p_in_t[0]  # A = x_hat/p_in
        B = desired[1]/state.p_in_t[0]  # B = y_hat/p_in
        Rl_subs = 2**(1/2)
        R_theor = state.p_in_t[0]*np.array([(1-A)/(A*(1+1/Rl_subs)-B/Rl_subs), (1-B)/(B*(1+1/Rl_subs)-A/Rl_subs)])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
    elif variabs.task_type == 'Regression':
        Rl_subs = 2**(1/2)
        R_theor = np.array([Rl_subs*(1-A-B)/A, Rl_subs*(1-A-B)/B])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
    elif variabs.task_type=='General_reg' or variabs.task_type=='General_reg_allRsChange':
        A = desired[0]/state.p_in_t[0]  # A = x_hat/p_in
        B = desired[1]/state.p_in_t[0]  # B = y_hat/p_in
#         C = desired[2]/state.p_in_t[0]  # B = y_hat/p_in
#         D = desired[3]/state.p_in_t[0]  # B = y_hat/p_in
#         E = desired[4]/state.p_in_t[0]  # B = y_hat/p_in
#         F = desired[5]/state.p_in_t[0]  # B = y_hat/p_in
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$', \
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$z\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$',
                   r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$', r'$R_6$']
        R_theor = []  # I didn't calculate it for this task
    elif variabs.task_type=='2by2' or variabs.task_type=='2by2_allRsChange':
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$']
        R_theor = []  # I didn't calculate it for this task
    legend4 = ['|loss|']
    print('R theoretical', R_theor)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.2))
    if variabs.task_type=='Allostery':
        ax1.plot(state.out_in_t[1:]/desired-1)
        ax1.plot(np.zeros(len(state.out_in_t[1:])), '--')
    # regression goes only every two samples
    else:
        ax1.plot(np.linspace(0, state.t, 2*state.t-1).T, \
                 np.asarray(state.out_in_t[1:])/np.asarray(state.desired_in_t[1:])-1)
#           ax1.plot(np.linspace(0, state.t, 2*state.t-1).T, \
#                    np.column_stack(state.out_in_t[1:])[0]/np.column_stack(state.desired_in_t[1:])[0] - 1)
    ax1.set_title('output in time')
    ax1.set_xlabel('t')
    ax1.legend(legend1)
    ax2.plot(state.out_dual_in_t[1:])
    ax2.plot(state.p_in_t[1:])
    ax2.set_title('dual and p in time')
    ax2.set_xlabel('t')
    # ax2.set_ylim([-0.2,0.2])
    ax2.legend(legend2)
    ax3.plot(state.R_in_t[1:])
    ax3.plot(np.outer(R_theor,np.ones(state.t)).T, '--')
    ax3.set_title('R in time')
    ax3.set_xlabel('t')
    ax3.legend(legend3)
    if variabs.task_type=='Allostery' or variabs.task_type=='Regression':  # loss is 2D in Allostery
        ax4.plot(np.abs(state.loss_in_t[1:]))
    elif variabs.task_type=='General_reg' or variabs.task_type=='General_reg_allRsChange':    
        ax4.plot(np.mean(np.mean(np.abs(state.loss_in_t[1:]), axis=1),axis=1))
    elif variabs.task_type=='2by2' or variabs.task_type=='2by2_allRsChange':    
        ax4.plot(np.mean(np.mean(np.abs(state.loss_in_t[1:]), axis=1),axis=1))    
    ax4.set_xlabel('t')
    ax4.legend(legend4)
    fig.suptitle(f'alpha={variabs.alpha}')
    plt.show()

 
def plotNetStructure(NET: nx.DiGraph, plot=False, node_labels=False) -> Dict[Any, Tuple[float, float]]:
    """
    Plots the structure (nodes and edges) of networkx NET 
    
    input:
    NET         - networkx net of nodes and edges
    plot        - 'yes'/'no', whether to plot or not
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


def PlotNetwork(p, u, K, BigClass, EIEJ_plots, NN, NE, nodes='yes', edges='yes', pressureSurf='no', savefig='no'):
    """
    Plots the flow network structure alongside hydrostatic pressure, flows and conductivities
    pressure denoted by colors from purple (high) to cyan (low)
    flow velocity denoted by arrow direction and thickness
    conductivity denoted by arrow color - black (low) and blue (high)
    
    input:
    p - 1D np.array sized [NNodes], hydrostatic pressure at nodes
    u - 1D np.array sized [NEdges], flow velocities at edges (positive is into cell center)
    K - 1D np.array sized [NEdges], flow conductivities for every edge
    pos_lattice - dict from
    layout - graph visual layout, string. Roie style is 'Cells'
    
    output:
    matplotlib plot of network structure
    """
    NET = BigClass.NET.NET
    pos_lattice = BigClass.NET.pos_lattice

    # Preliminaries for the plot
    node_sizes = 4*24
    u_rescale_factor = 5
    
    # p values at nodes - the same in EIEJ and in networkx's NET
    val_map = {i : p[i][0] for i in range(NN)}
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

    low_K_NET_inds = np.where(K_NET==BigClass.Variabs.K_min)[0]  # indices of edges with low conductivity
    high_K_NET_inds = np.where(K_NET!=BigClass.Variabs.K_min)[0]  # indices of edges with higher conductivity
    positive_u_NET_inds = np.where(u_NET>10**-10)[0]  # indices of edges with positive flow vel
    negative_u_NET_inds = np.where(u_NET<-10**-10)[0]  # indices of edges with negative flow vel
    
    r_edges_positive = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(positive_u_NET_inds))]  # edges with low conductivity, positive flow
    r_edges_negative = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(negative_u_NET_inds))]  # edges with low conductivity, negative flow
    b_edges_positive = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(positive_u_NET_inds))]  # edges with high conductivity, positive flow
    b_edges_negative = [NETEdges[i] for i in list(set(high_K_NET_inds) & set(negative_u_NET_inds))]  # edges with high conductivity, negative flow

    # save arrow sizes
    rescaled_u_NET = abs(u_NET)*u_rescale_factor/max(abs(u_NET))

    edgewidths_k_positive = rescaled_u_NET[list(set(low_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_k_negative = rescaled_u_NET[list(set(low_K_NET_inds) & set(negative_u_NET_inds))]
    edgewidths_b_positive = rescaled_u_NET[list(set(high_K_NET_inds) & set(positive_u_NET_inds))]
    edgewidths_b_negative = rescaled_u_NET[list(set(high_K_NET_inds) & set(negative_u_NET_inds))]

    if pressureSurf == 'yes':
        p_mat = statistics.p_mat(BigClass, p)
        figsizeX = 5*np.shape(p_mat)[0]
        figsizeY = 5*np.shape(p_mat)[1]
        X = np.arange(0, figsizeX, 5)
        Y = np.arange(0, figsizeY, 5)
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        plt.contourf(X, Y, p_mat, cmap=plt.cm.cool, linewidth=0, antialiased=False)

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    if nodes == 'yes':
        nx.draw_networkx_nodes(NET, pos_lattice, cmap=plt.get_cmap('cool'), 
                                node_color = values, node_size = node_sizes)
    
    if edges == 'yes':
        # draw with right arrow widths ("width") and directions ("arrowstyle")
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='r', arrows=True, width=edgewidths_k_positive,
                               arrowstyle='->')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_positive, edge_color='r', arrows=True, width=1,
                               arrowstyle='-[')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='r', arrows=True, width=edgewidths_k_negative,
                               arrowstyle='<-')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=r_edges_negative, edge_color='r', arrows=True, width=1,
                               arrowstyle=']-')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_positive, edge_color='k', arrows=True, width=edgewidths_b_positive,
                               arrowstyle='->')
        nx.draw_networkx_edges(NET, pos_lattice, edgelist=b_edges_negative, edge_color='k', arrows=True, width=edgewidths_b_negative,
                               arrowstyle='<-')
    if savefig=='yes':
        # prelims for figures
        comp_path = "C:\\Users\\SMR_Admin\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
        # comp_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\PhD\\Network Simulation repo\\figs and data\\"
        datenow = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        plt.savefig(comp_path + 'network_' + str(datenow) + '.png', bbox_s='tight')
    plt.show()


def PlotPressureContours(BigClass, p):
    """
    returns a matrix of pressures given some form of p, not in use...

    inputs:
    BigClass
    p

    outputs:
    p_mat
    """
    p_mat = Statistics.p_mat(BigClass, p)
    return p_mat