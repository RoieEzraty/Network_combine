from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy.typing import NDArray
from numpy import array, zeros, arange
from typing import TYPE_CHECKING

import solve

if TYPE_CHECKING:
    from Network_Structure import Network_Structure


############# functions that operate on matrices #############


def build_input_output_and_ground(Nin: int, Ninter: int, Nout: int) -> Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    build_input_output_and_ground builds the input and output pairs and ground node values

    inputs:
    task_type     - str, type of learning task the network should solve
                    'Allostery_one_pair' = 1 pair of input and outputs
                    'Allostery' = 2 pairs of input and outputs
                    'XOR' = 2 inputs and 2 outputs. difference between output nodes encodes the XOR result of the 2 inputs
                    'Channeling_diag' = 1st from input to diagonal output, then from output to 2 perpindicular nodes. 
                                        test from input to output
                    'Channeling_straight' = 1st from input to output on same column, then from output to 2 perpindicular 
                                            nodes. test from input to output (same as 1st)
                    'Counter' = column of cells bottomost and topmost nodes are input/output (switching), 
                                rightmost nodes (1 each row) ground. more about the task in "_counter.ipynb".
    sub_task_type - str, another specification of task, for regression whether there are 2 outputs or not etc.
    row           - int, # of row (and column) of cell in network from which input and output are considered
    NGrid         - int, row dimension of cells in network
    Nin           - int, # input nodes
    Nout          - int, # output nodes

    outputs:
    input_nodes_arr  - array of all input nodes in task 
    ground_nodes_arr - array of all output nodes in task
    output_nodes        - array of nodes with fixed values, for 'XOR' task. default=0
    """
    input_nodes_arr = array([i for i in range(Nin)])  # input nodes are first ones named
    inter_nodes_arr = array([Nin + i for i in range(Ninter)])  # output nodes are named later
    output_nodes_arr = array([Nin + Ninter + i for i in range(Nout)])  # output nodes are named later
    ground_nodes_arr = array([Nin + Ninter + Nout])  # last node is ground
    return input_nodes_arr, inter_nodes_arr, output_nodes_arr, ground_nodes_arr


def build_incidence(Strctr: "Network_Structure") -> Tuple[NDArray[np.int_], NDArray[np.int_], List[NDArray[np.int_]], NDArray[np.int_], int, int]:
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

    NN: int = len(Strctr.input_nodes_arr) + len(Strctr.output_nodes_arr) + len(Strctr.inter_nodes_arr) + 1
    ground_node: int = copy.copy(NN) - 1  # ground nodes is last one.
    EIlst = []
    EJlst = []

    # connect inputs to outputs
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, outNode in enumerate(Strctr.output_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(outNode)

    # connect input to inter
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, interNode in enumerate(Strctr.inter_nodes_arr):
            EIlst.append(inNode)
            EJlst.append(interNode)

    # connect inter to output
    for i, interNode in enumerate(Strctr.inter_nodes_arr):
        for j, outNode in enumerate(Strctr.output_nodes_arr):
            EIlst.append(interNode)
            EJlst.append(outNode)

    # connect input to ground
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        EIlst.append(inNode)
        EJlst.append(ground_node)

    # connect output to ground
    for i, outNode in enumerate(Strctr.output_nodes_arr):
        EIlst.append(outNode)
        EJlst.append(ground_node)
                
    EI = array(EIlst)
    EJ = array(EJlst)
    NE = len(EI)
            
    # for plots
    EIEJ_plots: list = [(EI[i], EJ[i]) for i in range(len(EI))]
    
    DM: np.ndarray = zeros([NE, NN])  # Incidence matrix
    for i in range(NE):
        DM[i,int(EI[i])] = +1.
        DM[i,int(EJ[i])] = -1.
        
    return EI, EJ, EIEJ_plots, DM, NE, NN


def buildL(BigClass, DM, K_mat, Cstr, NN):
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
    L = solve.dot_triple(DM.T, K_mat, DM)
    L_bar = zeros([NN + len(Cstr), NN + len(Cstr)])
    L_bar[NN:,:NN] = Cstr  # the bottom most rows of augmented L are the constraints
    L_bar[:NN,NN:] = Cstr.T  # the rightmost columns of augmented L are the constraints
    L_bar[:NN,:NN] = L  # The topmost and leftmost part of augmented L are the basic L
    return L, L_bar

def K_from_R(R_vec: np.ndarray, NE: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given resistances, calculate conductivities, output vector and matrix
    """
    K_vec = 1/R_vec
    K_mat = np.eye(NE)*K_vec
    return K_vec, K_mat

def ConstraintMatrix(NodeData, Nodes, GroundNodes, NN, EI, EJ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds constraint matrix, 
        for constraints on edge voltage drops: 1 at input node index, -1 at output and voltage drop at NN+1 index, for every row
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
        Cstr_full = 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges. last column is value of constraint
                 (p value of row contains just +1. pressure drop if row contains +1 and -1)
        Cstr      = 2D array without last column (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
        f         = constraint vector (from Rocks and Katifori 2018)
        """

        # ground nodes
        csg = len(GroundNodes)
        idg = arange(csg)
        CStr = zeros([csg, NN+1])
        CStr[idg, GroundNodes] = +1.
        CStr[:, NN] = 0.
        
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
        f[NN:,0] = CStr[:,-1]

        return CStr, CStr[:,:-1], f