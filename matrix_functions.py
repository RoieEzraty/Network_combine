from __future__ import annotations
import numpy as np
import copy

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


def build_input_output_and_ground(Nin: int, extraNin: int, Ninter: int,
                                  Nout: int, extraNout: int) -> Tuple[NDArray[np.int_], NDArray[np.int_],
                                                                      NDArray[np.int_], NDArray[np.int_],
                                                                      NDArray[np.int_], NDArray[np.int_],]:
    """
    build_input_output_and_ground builds the input and output pairs and ground node values as arrays

    inputs:
    Nin    - int, # input nodes
    Ninter - int, # intermediate nodes between input and output
    Nout   - int, # output nodes

    outputs:
    input_nodes_arr  - array of all input nodes in task
    inter_nodes_arr  - array of all intermediate nodes in task between input and output
    ground_nodes_arr - array of all output nodes in task
    output_nodes     - array of nodes with fixed values, for 'XOR' task. default=0
    """
    input_nodes_arr: NDArray[np.int_] = array([i for i in range(Nin)])  # input nodes are first ones named
    # extra inputs not accounted in loss
    extraInputs_nodes_arr: NDArray[np.int_] = array([Nin + i for i in range(extraNin)], dtype=np.int_)
    inter_nodes_arr: NDArray[np.int_] = array([Nin + extraNin + i for i in range(Ninter)])  # intermediate nodes
    output_nodes_arr: NDArray[np.int_] = array([Nin + extraNin + Ninter + i for i in range(Nout)])  # output nodes
    # extra outputs not accounted in loss
    extraOutput_nodes_arr: NDArray[np.int_] = array([Nin + extraNin + Ninter + Nout + i for i in range(extraNout)],
                                                    dtype=np.int_)
    ground_nodes_arr: NDArray[np.int_] = array([Nin + extraNin + Ninter + Nout + extraNout])  # last node is ground
    inInterOutGround_tuple = (input_nodes_arr, extraInputs_nodes_arr, inter_nodes_arr, output_nodes_arr,
                              extraOutput_nodes_arr, ground_nodes_arr)
    return inInterOutGround_tuple


def build_incidence(Strctr: "Network_Structure") -> Tuple[NDArray[np.int_], NDArray[np.int_], List[NDArray[np.int_]],
                                                          NDArray[np.int_], int, int]:
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
        len(Strctr.output_nodes_arr) + len(Strctr.extraInput_nodes_arr) + 1
    ground_node: int = copy.copy(NN) - 1  # ground nodes is last one.
    EIlst: List[int] = []
    EJlst: List[int] = []

    # connect inputs to outputs
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, outNode in enumerate(Strctr.output_nodes_arr):
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

    # connect input to ground
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        EIlst.append(inNode)
        EJlst.append(ground_node)

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


def buildL(BigClass: "Big_Class", DM: NDArray[np.int_], K_mat: NDArray[np.float_], Cstr: NDArray[np.float_],
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


def K_from_R(R_vec: NDArray[np.float_], NE: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given resistances, calculate conductivities, output vector and matrix

    inputs:
    R_vec - resistances as array sized [NE,]
    NE    - # edges, int

    outputs:
    K_vec - conductances as array sized [NE,]
    K_vec - conductances as array sized [NE, NE] with off diagonal element = 0
    """
    K_vec: NDArray[np.float_] = 1/R_vec
    # Replace -inf with a large negative value (or directly clip it)
    K_vec = np.nan_to_num(K_vec, nan=0.0, posinf=1e+06, neginf=-1e+06)
    K_mat: NDArray[np.float_] = np.eye(NE)*K_vec
    return K_vec, K_mat


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


def edges_from_EI_EJ(nodes_array, EI, EJ) -> NDArray[np.int_]:
    """
    add descrpt
    """
    edges: NDArray[np.int_] = array([np.where(np.append(EI, EJ) == nodes_array[i])[0] % len(EI)
                                     for i in range(len(nodes_array))])
    return edges
