import numpy as np
import copy
from numpy import zeros as zeros
from numpy import ones as ones
from numpy import array as array
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

import statistics


# ===================================================
# functions that operate on netowrkx networks
# ===================================================


def plotNetStructure(NET, NGrid, scale, squish, layout='Cells', plot='no', node_labels=True):
    """
    Plots the structure (nodes and edges) of networkx NET

    input:
    NET         - networkx net of nodes and edges
    NGrid       - int, # cells in each dimensions (vert, horiz), relevant only for 'Cells' layour
    scale       - float, spread between cells - the bigger, the more spread out
    squish      - float, spread of edges within each cell - the bigger, the more spread out
    layout      - str graph visual layout, string. Roie style is 'Cells'
    plot        - 'yes'/'no', whether to plot or not
    node_labels - boolean, show node number in plot or not

    output:
    pos_lattice - dict of positions of nodes from NET.nodes
    if plot=='yes' also show matplotlib plot of network structure
    """
    if layout == 'Cells':  # Roie style of 2D array of connected crosses
        pos_lattice = {}  # initiate dictionary of node positions
        # NGrid = int(np.sqrt(len(NET.nodes)/5))  # number of cells in network, considering only cubic array of cells
        k = 0  # dummy
        for i in range(NGrid):  # network rows
            for j in range(NGrid):  # network columns
                pos_lattice[scale*(i+j+k)] = array([-(scale/2-squish)+scale*j, 0+scale*i])  # left node in cell
                pos_lattice[scale*(i+j+k)+1] = array([0+scale*j, -(scale/2-squish)+scale*i])  # lower node in cell
                pos_lattice[scale*(i+j+k)+2] = array([(scale/2-squish)+scale*j, 0+scale*i])  # right node
                pos_lattice[scale*(i+j+k)+3] = array([0+scale*j, (scale/2-squish)+scale*i])  # upper node
                pos_lattice[scale*(i+j+k)+4] = array([0+scale*j, 0+scale*i])  # middle node
            k += NGrid-1  # add to dummy index so skipping to next cell
    elif layout == 'oneCol':  # Roie style of single column of connected crosses
        pos_lattice = {}  # initiate dictionary of node positions
        # NGrid = int(len(NET.nodes)/5)  # number of cells in network, considering only cubic array of cells
        for i in range(NGrid):  # network columns
            pos_lattice[scale*i] = array([-(scale/2-squish), 0+scale*i])  # left node in cell
            pos_lattice[scale*i+1] = array([0, -(scale/2-squish)+scale*i])  # lower node in cell
            pos_lattice[scale*i+2] = array([(scale/2-squish), 0+scale*i])  # right node
            pos_lattice[scale*i+3] = array([0, (scale/2-squish)+scale*i])  # upper node
            pos_lattice[scale*i+4] = array([0, 0+scale*i])  # middle node
    elif layout == 'spectral':
        pos_lattice = nx.spectral_layout(NET)
    elif layout == 'planar':
        pos_lattice = nx.planar_layout(NET)
    elif layout == 'FC':
        pos_lattice = nx.spring_layout(NET)
    else:
        pos_lattice = nx.spectral_layout(NET)

    if plot == 'yes':
        nx.draw_networkx(NET, pos_lattice, edge_color='b', node_color='b', with_labels=node_labels)
        # nx.draw_networkx(NET, pos_lattice, edge_color='b', node_color='b', with_labels=False)
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

    low_K_NET_inds = np.where(K_NET == BigClass.Variabs.K_min)[0]  # indices of edges with low conductivity
    high_K_NET_inds = np.where(K_NET != BigClass.Variabs.K_min)[0]  # indices of edges with higher conductivity
    positive_u_NET_inds = np.where(u_NET > 10**-10)[0]  # indices of edges with positive flow vel
    negative_u_NET_inds = np.where(u_NET < -10**-10)[0]  # indices of edges with negative flow vel

    r_edges_positive = [NETEdges[i] for i in list(set(low_K_NET_inds) & set(positive_u_NET_inds))]  # low conductivity edges,
                                                                                                    # edges, positive flow
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
        p_mat = Statistics.p_mat(BigClass, p)
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
