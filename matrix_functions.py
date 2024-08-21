from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Network_Structure import Network_Structure


############# functions that operate on matrices #############


def build_input_output_and_ground(Nin: int, Nout: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    ground_noides_arr - array of all output nodes in task
    output_nodes        - array of nodes with fixed values, for 'XOR' task. default=0
    """
    input_nodes_arr = array([i for i in range(Nin)])  # input nodes are first ones named
    output_nodes_arr = array([Nin + i for i in range(Nout)])  # output nodes are named later
    ground_noides_arr = array([Nin + Nout])  # last node is ground
    return input_nodes_arr, ground_noides_arr, output_nodes_arr


def build_incidence(Strctr: "Network_Structure") -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, int, int]:
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes]
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    a        - N cells in vertical direction of lattice, int
    b        - N cells in horizontal direction of lattice, int
    typ      - type of lattice (Nachi style or mine) str
    Periodic - 1 if lattice is periodic, 0 if not

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """

    NN: int = len(Strctr.input_nodes_arr) + len(Strctr.output_nodes_arr) + 1
    ground_node: int = copy.copy(NN) - 1
    EIlst = []
    EJlst = []

    # connect inputs to outputs
    for i, inNode in enumerate(Strctr.input_nodes_arr):
        for j, outNode in enumerate(Strctr.output_nodes_arr):
            EIlst.append(inNode)
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
    print('EI', EI)
    print('EJ', EJ)
    print('NE', NE)
            
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
    L = BigClass.Solver.solve.dot_triple(DM.T, K_mat, DM)
    L_bar = zeros([NN + len(Cstr), NN + len(Cstr)])
    L_bar[NN:,:NN] = Cstr  # the bottom most rows of augmented L are the constraints
    L_bar[:NN,NN:] = Cstr.T  # the rightmost columns of augmented L are the constraints
    L_bar[:NN,:NN] = L  # The topmost and leftmost part of augmented L are the basic L
    return L, L_bar