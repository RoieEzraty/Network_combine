from __future__ import annotations
import numpy as np
import copy
import itertools
import networkx as nx
import random as rand

from typing import Tuple, List
from numpy.typing import NDArray
from numpy import array, zeros, arange
from typing import TYPE_CHECKING

import solve

if TYPE_CHECKING:
    from Network_Structure import Network_Structure
    from Big_Class import Big_Class


# ===================================================
# functions that operate on matrices
# ===================================================


def build_input_output_and_ground(Nin: int, Nout: int, in_nodes: NDArray[np.int_] = array([]),
                                  out_nodes: NDArray[np.int_] = array([]), add_ground=True, net_type="FC", seed=42,
                                  net_height: int = 0, net_len: int = 0, extraNin: int = 0, Ninter: int = 0,
                                  extraNout: int = 0) -> Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                                               NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    build_input_output_and_ground builds the input and output pairs and ground node values as arrays

    inputs:
    Nin        - int, # input nodes
    Ninter     - int, # intermediate nodes between input and output
    Nout       - int, # output nodes
    out_nodes  - optional NDArray of indices of input nodes, otherwise decided here
    out_nodes  - optional NDArray of indices of output nodes, otherwise decided here
    add_ground - optional boolean of whether to add ground node as last one
    net_type   - optional string, type of network structure:
                 "FC" - Fully Connected, each input connected to each output (and ground)
                 "square" - 2D square network, each node connected to additional 4
                 "beads" - crosses where resistances are binary low/high
    seed       - int, random seed choosing node index of input output etc. for square network
    net_height - int, number of rows square network
    net_len    - int, number of columns square network
    extranNin  - int, number of extra input nodes (of which loss is not calculated)
    Ninter     - int, number of nodes that are between input and output, floating
    extraNout  - int, number of extra output nodes (of which loss is not calculated)

    outputs:
    input_nodes_arr  - array of all input nodes in task
    inter_nodes_arr  - array of all intermediate nodes in task between input and output
    ground_nodes_arr - array of all output nodes in task
    output_nodes     - array of nodes with fixed values, for 'XOR' task. default=0
    """
    if net_type == "square":
        if add_ground:
            Nground: int = 1
        else:
            Nground = 0
        rand.seed(seed)
        if net_height * net_len == 2:  # pathologically small net
            rand_nodes: list[int] = [0, 1]
        elif net_height * net_len == 3:
            rand_nodes = [0, 2, 1]
        elif net_height * net_len == 4:
            rand_nodes = [0, 2]
        else:  # normal net
            rand_nodes = rand.sample(range(0, net_height * net_len),
                                     Nin + extraNin + Ninter + Nout + extraNout + Nground)
        # input nodes
        if in_nodes.size > 0:  # input nodes assigned by user
            input_nodes_arr: NDArray[np.int_] = in_nodes
        else:
            input_nodes_arr = array([rand_nodes[i] for i in range(Nin)], dtype=np.int_)
        # extra inputs not accounted in loss
        extraInputs_nodes_arr: NDArray[np.int_] = array([rand_nodes[Nin + i] for i in range(extraNin)], dtype=np.int_)
        # intermediate nodes
        inter_nodes_arr: NDArray[np.int_] = array([rand_nodes[Nin + extraNin + i] for i in range(Ninter)],
                                                  dtype=np.int_)
        # output nodes
        if out_nodes.size > 0:  # output nodes assigned by user
            output_nodes_arr: NDArray[np.int_] = out_nodes
        else:
            output_nodes_arr = array([rand_nodes[Nin+i] for i in range(Nout)], dtype=np.int_)
        # extra outputs
        extraOutput_nodes_arr: NDArray[np.int_] = array([rand_nodes[Nin + extraNin + Ninter + Nout + i]
                                                        for i in range(extraNout)], dtype=np.int_)
        if add_ground:
            # last node is ground
            ground_nodes_arr: NDArray[np.int_] = array([rand_nodes[Nin + Nout]], dtype=np.int_)
        else:  # don't add a ground node where p=0
            ground_nodes_arr = array([], dtype=np.int_)
    elif net_type == "beads":
        row = 1
        input_nodes_arr = array([(row*net_height+(row+1))*5-1])
        extraInputs_nodes_arr = array([], dtype=np.int_)
        inter_nodes_arr = array([], dtype=np.int_)
        if add_ground:
            ground_nodes_arr = array([(net_height*(net_len-row)-row)*5-1])
        else:
            ground_nodes_arr = array([], dtype=np.int_)
        extraOutput_nodes_arr = array([], dtype=np.int_)
        output_nodes_arr = array([((row+1)*net_height-row)*5-1])
    else:  # network is Fully Connected ("FC")
        # input nodes
        input_nodes_arr = array([i for i in range(Nin)])  # input nodes are first ones named
        # extra inputs not accounted in loss
        extraInputs_nodes_arr = array([Nin + i for i in range(extraNin)], dtype=np.int_)
        # intermediate nodes
        inter_nodes_arr = array([Nin + extraNin + i for i in range(Ninter)], dtype=np.int_)
        # output nodes
        output_nodes_arr = array([Nin + extraNin + Ninter + i for i in range(Nout)])
        # extra outputs not accounted in loss
        extraOutput_nodes_arr = array([Nin + extraNin + Ninter + Nout + i for i in range(extraNout)], dtype=np.int_)
        if add_ground:
            # last node is ground
            ground_nodes_arr = array([Nin + extraNin + Ninter + Nout + extraNout])
        else:  # don't add a ground node where p=0
            ground_nodes_arr = array([], dtype=np.int_)
    # put all in tuple
    inInterOutGround_tuple = (input_nodes_arr, extraInputs_nodes_arr, inter_nodes_arr, output_nodes_arr,
                              extraOutput_nodes_arr, ground_nodes_arr)
    return inInterOutGround_tuple


def build_incidence(Strctr: "Network_Structure") -> Tuple[NDArray[np.int_], NDArray[np.int_], List[NDArray[np.int_]],
                                                          NDArray[np.int_], int, int]:
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes] for 1 single FC network, w/out ground
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    Strctr: "Network_Structure" class instance with the input, intermediate and output nodes

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """

    NN: int = len(Strctr.input_nodes_arr) + len(Strctr.extraInput_nodes_arr) + len(Strctr.inter_nodes_arr) + \
        len(Strctr.output_nodes_arr) + len(Strctr.extraOutput_nodes_arr) + len(Strctr.ground_nodes_arr)
    ground_node: int = copy.copy(NN) - 1  # ground nodes is last one.
    EIlst: List[int] = []
    EJlst: List[int] = []

    # # connect inputs to outputs
    # for i, inNode in enumerate(Strctr.input_nodes_arr):
    #     for j, outNode in enumerate(Strctr.output_nodes_arr):
    #         EIlst.append(inNode)
    #         EJlst.append(outNode)
    # connect inputs to outputs ONLY IF no intermediate nodes exist
    if len(Strctr.inter_nodes_arr) == 0:
        for inNode in Strctr.input_nodes_arr:
            for outNode in Strctr.output_nodes_arr:
                EIlst.append(inNode)
                EJlst.append(outNode)

    # connect inputs to extraOutputs
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, outNode in enumerate(Strctr.extraOutput_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(outNode)

    # connect input to inter
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, interNode in enumerate(Strctr.inter_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(interNode)

    # connect extraInputs to outputs
    for i, inNode in enumerate(Strctr.extraInput_nodes_arr):
        for j, outNode in enumerate(Strctr.output_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(outNode)

    # connect extraInputs to extraOutputs
    for i, inNode in enumerate(Strctr.extraInput_nodes_arr):
        for j, outNode in enumerate(Strctr.extraOutput_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(outNode)

    # connect extraInputs to inter
    for i, inNode in enumerate(Strctr.extraInput_nodes_arr):
        for j, interNode in enumerate(Strctr.inter_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(interNode)

    # connect inter to output
    for i, interNode in enumerate(Strctr.inter_nodes_arr):
        for j, outNode in enumerate(Strctr.output_nodes_arr):
            EIlst.append(interNode)
            EJlst.append(outNode)

    # connect inter to extraOutput
    for i, interNode in enumerate(Strctr.inter_nodes_arr):
        for j, outNode in enumerate(Strctr.extraOutput_nodes_arr):
            EIlst.append(interNode)
            EJlst.append(outNode)

    # # connect input to ground
    # for i, inNode in enumerate(Strctr.input_nodes_arr):
    #     EIlst.append(inNode)
    #     EJlst.append(ground_node)

    # connect extraInput to ground
    for i, inNode in enumerate(Strctr.extraInput_nodes_arr):
        EIlst.append(inNode)
        EJlst.append(ground_node)

    # connect output to ground
    for i, outNode in enumerate(Strctr.output_nodes_arr):
        EIlst.append(outNode)
        EJlst.append(ground_node)

    # connect extraOutput to ground
    for i, outNode in enumerate(Strctr.extraOutput_nodes_arr):
        EIlst.append(outNode)
        EJlst.append(ground_node)

    if Strctr.net_type == 'FC_connected_outputs':  # connect all outputs between themselves
        # Generate unique pairs and store them in separate lists
        for i, j in itertools.combinations(Strctr.output_nodes_arr, 2):
            EIlst.append(i)
            EJlst.append(j)

    EI: NDArray[np.int_] = array(EIlst)
    EJ: NDArray[np.int_] = array(EJlst)
    NE: int = len(EI)

    # for plots
    EIEJ_plots: List = [(EI[i], EJ[i]) for i in range(len(EI))]

    DM: NDArray[np.int_] = zeros([NE, NN], dtype=np.int_)  # Incidence matrix
    for i in range(NE):
        DM[i, int(EI[i])] = +1.
        DM[i, int(EJ[i])] = -1.

    return EI, EJ, EIEJ_plots, DM, NE, NN


def build_incidence_partialInter(Strctr: "Network_Structure") -> Tuple[NDArray[np.int_], NDArray[np.int_],
                                                                       List[NDArray[np.int_]], NDArray[np.int_],
                                                                       int, int]:
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes]
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    Strctr: "Network_Structure" class instance with the input, intermediate and output nodes

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """

    NN: int = len(Strctr.input_nodes_arr) + len(Strctr.extraInput_nodes_arr) + len(Strctr.inter_nodes_arr) + \
        len(Strctr.output_nodes_arr) + len(Strctr.extraOutput_nodes_arr) + 1
    ground_node: int = copy.copy(NN) - 1  # ground nodes is last one.
    EIlst: List[int] = []
    EJlst: List[int] = []

    # connect input to inter
    k = 0
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, outputNode in enumerate(Strctr.output_nodes_arr):
            interNode = Strctr.inter_nodes_arr[k]
            EIlst.append(inNode)
            EJlst.append(interNode)
            EIlst.append(interNode)
            EJlst.append(outputNode)
            k += 1

    # connect input to ground
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        EIlst.append(inNode)
        EJlst.append(ground_node)

    # connect output to ground
    for i, outNode in enumerate(Strctr.output_nodes_arr):
        EIlst.append(outNode)
        EJlst.append(ground_node)

    EI: NDArray[np.int_] = array(EIlst)
    EJ: NDArray[np.int_] = array(EJlst)
    NE: int = len(EI)

    # for plots
    EIEJ_plots: List = [(EI[i], EJ[i]) for i in range(len(EI))]

    DM: NDArray[np.int_] = zeros([NE, NN], dtype=np.int_)  # Incidence matrix
    for i in range(NE):
        DM[i, int(EI[i])] = +1.
        DM[i, int(EJ[i])] = -1.

    return EI, EJ, EIEJ_plots, DM, NE, NN


def build_incidence_square(Strctr: "Network_Structure") -> Tuple[NDArray[np.int_], NDArray[np.int_],
                                                                 List[NDArray[np.int_]], NDArray[np.int_], int, int]:
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes] for a square network
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    Strctr: "Network_Structure" class instance with the input, intermediate and output nodes

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """

    NN: int = Strctr.net_height*Strctr.net_len
    SQRENET = nx.grid_2d_graph(Strctr.net_height, Strctr.net_len, periodic=False, create_using=None)
    EIlst: List[int] = []
    EJlst: List[int] = []

    for (x1, y1), (x2, y2) in SQRENET.edges:
        EIlst.append(y1 * Strctr.net_height + x1)
        EJlst.append(y2 * Strctr.net_height + x2)

    EI: NDArray[np.int_] = array(EIlst)
    EJ: NDArray[np.int_] = array(EJlst)
    NE: int = len(EI)

    # for plots
    EIEJ_plots: List = [(EI[i], EJ[i]) for i in range(len(EI))]

    DM: NDArray[np.int_] = zeros([NE, NN], dtype=np.int_)  # Incidence matrix
    for i in range(NE):
        DM[i, int(EI[i])] = +1.
        DM[i, int(EJ[i])] = -1.

    return EI, EJ, EIEJ_plots, DM, NE, NN


def build_incidence_beads(Strctr: "Network_Structure") -> Tuple[NDArray[np.int_], NDArray[np.int_],
                                                                List[NDArray[np.int_]], NDArray[np.int_], int, int]:
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes] for a square network
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    Strctr: "Network_Structure" class instance with the input, intermediate and output nodes

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """

    # 5 nodes in each cross in 2D array
    NN: int = 5*Strctr.net_height*Strctr.net_len

    EIlst: List[int] = []
    EJlst: List[int] = []

    # The cells individually
    for i in range(Strctr.net_height*Strctr.net_len):
        for j in range(4):
            EIlst.append(5*i+j)
            EJlst.append(5*i+4)

    # Connecting them
    for i in range(Strctr.net_height-1):
        for j in range(Strctr.net_height-1):
            EIlst.append(5*i*Strctr.net_height + 5*j + 2)
            EJlst.append(5*i*Strctr.net_height + 5*j + 5)
            EIlst.append(5*i*Strctr.net_height + 5*j + 3)
            EJlst.append(5*(i+1)*Strctr.net_height + 5*j + 1)
        EIlst.append(5*(i+1)*Strctr.net_height - 2)
        EJlst.append(5*(i+2)*Strctr.net_height - 4)
    for j in range(Strctr.net_height-1):
        EIlst.append(5*(Strctr.net_height-1)*Strctr.net_len + 5*j + 2)
        EJlst.append(5*(Strctr.net_height-1)*Strctr.net_len + 5*j + 5)

    EI: NDArray[np.int_] = array(EIlst)
    EJ: NDArray[np.int_] = array(EJlst)
    NE: int = len(EI)

    # for plots
    EIEJ_plots: List = [(EI[i], EJ[i]) for i in range(len(EI))]

    print('EI', EI)
    print('EJ', EJ)

    DM: NDArray[np.int_] = zeros([NE, NN], dtype=int)  # Incidence matrix
    for i in range(NE):
        DM[i, int(EI[i])] = +1.
        DM[i, int(EJ[i])] = -1.

    return EI, EJ, EIEJ_plots, DM, NE, NN


def inverse_incidence(DM: NDArray[np.int_]) -> NDArray[np.float_]:
    """
    inverts incidence matrix, should be done once for GD-like scheme

    input:
    DM - Incidence matrix np.array [NE, NN]

    output:
    DM_dagger - Shortened Lagrangian np.array cubic array sized [NNodes]
    """
    return np.linalg.pinv(DM)


def build_rep_sel(Nin: int, Nout: int, DM: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Build repetition-selection matrix for dp^!=RM*(y_hat-y)/(DM*x)
    """
    # Build R: ???
    RM = np.zeros(np.shape(DM), dtype=int)
    for j in range(np.shape(DM)[0]):
        RM[j, DM[j, :] == -1] = 1
    return RM


def buildL(DM: NDArray[np.int_], K_mat: NDArray[np.float_], Cstr: NDArray[np.float_],
           NN: int) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Builds expanded Lagrangian with constraints
    as in the Methods section of Rocks and Katifori 2018 (https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    np.array cubic array sized [NNodes + Constraints]

    input:
    BigClass - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET)
               and network state (State) class instances
               I will not go into everything used from there to save space here.
    DM       - Incidence matrix np.array [NE, NN]
    K_mat    - cubic np.array sized NE with flow conductivities on diagonal
    Cstr     - np.array sized [Constraints, NN + 1] of constraints
    NN       - NNodes, ind

    output:
    L     - Shortened Lagrangian np.array cubic array sized [NNodes]
    L_bar - Full  augmented Lagrangian, np.array cubic array sized [NNodes + Constraints]
    """
    L: NDArray[np.float_] = solve.dot_triple(DM.T, K_mat, DM)
    L_bar: NDArray[np.float_] = zeros([NN + len(Cstr), NN + len(Cstr)])
    L_bar[NN:, :NN] = Cstr  # the bottom most rows of augmented L are the constraints
    L_bar[:NN, NN:] = Cstr.T  # the rightmost columns of augmented L are the constraints
    L_bar[:NN, :NN] = L  # The topmost and leftmost part of augmented L are the basic L
    return L, L_bar


def K_from_R(R_vec: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Given resistances, calculate conductances

    inputs:
    R_vec - resistances as array sized [NE,]

    outputs:
    K_vec - conductances as array sized [NE,]
    """
    K_vec: NDArray[np.float_] = 1/R_vec
    # Replace -inf with a large negative value (or directly clip it)
    K_vec = np.nan_to_num(K_vec, nan=0.0, posinf=1e+06, neginf=-1e+06)
    return K_vec


def ChangeRFromFlow(BigClass: "Big_Class", R_max, R_min, R_change_scheme='beads_pressure',
                    allowed_cells=[], beta=0.0):
    """
    Change conductivities of full network given velocities - for beads network only
    Dividing the network into cells and changing the K's cell by cell

    input:
    BigClass - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET)
           and network state (State) class instances
           I will not go into everything used from there to save space here.
    R_max           - float, value of maximal resistance
    R_min           - float, value of minimal resistance
    R_change_scheme - str, scheme for how to change conductivities due to u or p. default='beads_pressure'
    allowed_cells   - np.array of ints denoting the cells whose K's are allowed to change
    beta            - float, vaule for conductivity change proportional to velocity squared, default=0.0

    output:
    R_nxt - [NEdges] 1D  np.array of resistances for next iteration
    """
    u = BigClass.State.u  # flow velocity
    thresh = BigClass.Variabs.p_thresh  # threshold pressure to move bead
    R = BigClass.State.R_in_t[-1]
    R_nxt = copy.copy(R)
    R_backg = BigClass.State.R_backg  # default resistance configuration for all network

    if R_change_scheme == 'propto_current_squared':  # resistances (1/conductances) are proportional to Q^2
        u_sqrd_mean = np.mean(u**2)
        R_nxt = R + beta * u ** 2 / u_sqrd_mean * (R_max - R) / R_max
    # if conductances change due to delta p or Q
    elif R_change_scheme in ['beads_u', 'beads_pressure', 'beads_p_lower_l_half', 'beads_p_upper_l_half']:
        NCells = int(len(R_nxt)/4)  # total number of cells in network
        for i in range(NCells):  # change R's in every cell separately
            # skip update of the R's in that cell since it is not at lower left half of domain
            if (R_change_scheme in ['beads_p_lower_l_half', 'beads_p_upper_l_half']) and i not in allowed_cells:
                print(f'cell #{i} skipped')
                pass
            else:  # update R's cell by cell
                u_sub = u[4*i:4*(i+1)]  # velocities at particular cell
                R_sub = R[4*i:4*(i+1)]  # conductivities at particular cell
                if isinstance(thresh, np.ndarray):
                    thresh_sub = thresh[4*i:4*(i+1)]
                else:
                    thresh_sub = copy.copy(thresh)
                R_backg_sub = R_backg[4*i:4*(i+1)]  # background resistances at particular cell
                # change R's at particular cell
                R_sub_nxt = ChangeRFromFlow_singleCell(u_sub, thresh_sub, R_sub, R_backg_sub, R_max, R_min,
                                                       R_change_scheme)
                R_nxt[4*i:4*(i+1)] = R_sub_nxt  # put them in the right place at R_nxt
    return R_nxt


def ChangeRFromFlow_singleCell(u, p_thresh, R, R_backg, R_max, R_min, R_change_scheme):
    """
    Change conductivities of cell as a 2D cubic np.array sized 4
    u and K are sub vectors and matrices w/4 elements representing 4 edges of single cell

    input:
    u               - 1D np.array of flow through cell edges, 4 elements
    thresh          - threshold of velocity that moves the bead and changes K, float
    R               - [2, 2] 2D cubic np.array of conductivities
    R_backg         - [2, 2] 2D cubic np.array of background conductivities, as if no beads
    R_max           - value of maximal conductivity
    R_min           - value of minimal conductivity
    R_change_scheme - str, scheme for how to change conductivities due to u or p. default='beads_pressure'

    output:
    R_nxt - 2D cubic array of conductivities with 4 elements on diag for next iteration
    """

    # if beads move due to pressure difference delta_p
    if R_change_scheme in ['beads_pressure', 'beads_p_lower_l_half', 'beads_p_upper_l_half']:
        delta_p = u * R  # pressure difference at edge
        # all indices where u enters the cell at delta_p greater than threshold to move bead
        u_in_ind = np.where(delta_p > p_thresh)[0]
        u_out_ind = np.where(u == min(u.T))[0]  # indices if minimal flow, possibly exiting the cell
    elif R_change_scheme == 'beads_u':  # beads move due to flow u
        u_thresh = p_thresh / R_min
        # all indices where u enters the cell at velocity greater than threshold to move bead
        u_in_ind = np.where(u > u_thresh)[0]
        u_out_ind = np.where(u == min(u.T))[0]  # indices if minimal flow, possibly exiting the cell

    R_nxt = copy.copy(R_backg)
    # no flow exits the cell, it is a ground, don't put bead inside, else
    if not all(u[u_out_ind] > 0):  # normal cell, not ground
        # if two edges have exactly the same output flow, choose random one
        pick_u_out = [rand.choice(u_out_ind)]
        # check if there is flow inwards in edge where bead is at. then it has to move
        cond1 = len(list(set(np.where(R == R_max)[0]) & set(u_in_ind))) > 0
        cond2 = all(R < R_max)  # bead is in middle of cell (happens only at first simulation iteration)
        cond3 = len(u_in_ind) != 0  # inflow too weak to move bead
        # if flow moves bead from edge (1st cond) / middle (2nd cond), put lowest R there
        if (cond1 or cond2) and cond3:
            R_nxt[pick_u_out] = R_max
        else:  # flow does not change conductivity
            R_nxt = copy.copy(R)
    return R_nxt


def ConstraintMatrix(NodeData, Nodes, GroundNodes, NN, EI, EJ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds constraint matrix,
    For constraints on node voltages: 1 at constrained node index, voltage at NN+1 index, for every row
    For ground nodes: 1 at ground node index, 0 else.

    Inputs:
    NodeData    = 1D array at length as "Nodes" corresponding to pressures at each node from "Nodes"
    Nodes       = 1D array of nodes that have a constraint
    GroundNodes = 1D array of nodes that have a constraint of ground (outlet)
    NN          = int, number of nodes in network
    EI          = 1D array of nodes at each edge beginning
    EJ          = 1D array of nodes at each edge ending corresponding to EI

    outputs:
    Cstr_full = 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges.
                last column is value of constraint
                (p value of row contains just +1. pressure drop if row contains +1 and -1)
    Cstr      = 2D array without last column
                (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    f         = constraint vector (from Rocks and Katifori 2018)
    """

    # ground nodes
    csg = len(GroundNodes)
    idg = arange(csg)
    CStr = zeros([csg, NN+1])
    CStr[idg, GroundNodes] = +1.
    CStr[:, NN] = 0

    # constrained node pressures
    if len(Nodes):
        csn = len(Nodes)
        idn = arange(csn)
        SN = zeros([csn, NN+1])
        SN[idn, Nodes] = +1.
        SN[:, NN] = NodeData
        CStr = np.r_[CStr, SN]

    # to not lose functionality in the future if I want to add Edges as well
    Edges = array([])
    EdgeData = array([])

    # constrained edge pressure drops
    if len(Edges):
        cse = len(Edges)
        ide = arange(cse)
        SE = zeros([cse, NN+1])
        SE[ide, EI[Edges]] = +1.
        SE[ide, EJ[Edges]] = -1.
        SE[:, NN] = EdgeData
        CStr = np.r_[CStr, SE]

    # last column of CStr is vector f
    f = zeros([NN + len(CStr), 1])
    f[NN:, 0] = CStr[:, -1]

    return CStr, CStr[:, :-1], f
