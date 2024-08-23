from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List, Union
from numpy.typing import NDArray
from numpy import array, zeros, arange

import matrix_functions


############# other functions #############


def loss_fn_2samples(output1: NDArray[np.float_], output2: NDArray[np.float_], desired1: NDArray[np.float_], desired2: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    loss functions for regression task out=M*in using two sampled input pressures

    inputs:
    output1: np.ndarray sized [Nout,] output for 1st sample (current time step)
    output2: np.ndarray sized [Nout,] output for 2nd sample (previous time step)
    desired1: np.ndarray sized [Nout,] desired output for 1st sample (current time step)
    desired2: np.ndarray sized [Nout,] desired output for 2nd sample (previous time step)

    outputs:
    loss: np.ndarray sized [Nout, 2] loss as linear difference between output and desired, each line for a different sample
    """
    L1: np.ndarray = desired1-output1
    L2: np.ndarray = desired2-output2
    loss = np.array([L1, L2])
    return loss


def loss_fn_1sample(output: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """
    loss functions for regression task out=M*in using a single drawn input pressure

    inputs:
    output1: np.ndarray sized [Nout,] output for 1st sample (current time step)
    desired1: np.ndarray sized [Nout,] desired output for 1st sample (current time step)

    outputs:
    loss: np.ndarray sized [Nout,] loss as linear difference between output and desired, each line for a different sample
    """
    L1: np.ndarray = desired-output
    loss: np.ndarray = np.array([L1])
    return loss


def setup_constraints_given_pin(nodes_tuple: Union[Tuple[NDArray[np.int_], NDArray[np.int_]], Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]], nodeData_tuple: Union[Tuple[NDArray[np.float_], NDArray[np.float_]], NDArray[np.float_]],\
                                NN: int, EI: NDArray[np.int_], EJ: NDArray[np.int_]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build_incidence builds the incidence matrix DM

    inputs:
    BigClass: class instance consisting User_Variables, Network_State, etc.
    problem: str, "measure" for pressure posed on inputs and ground, "dual" for pressure also on outputs and change of resistances 

    outputs:
    Cstr_full = 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges. last column is value of constraint
                (p value of row contains just +1. pressure drop if row contains +1 and -1)
    Cstr      = 2D array without last column (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    f         = constraint vector (from Rocks and Katifori 2018)
    """
    # specific constraints for training step 
    NodeData, Nodes, GroundNodes = Constraints_nodes(nodes_tuple, nodeData_tuple)

    # print('NodeData', NodeData)
    # print('Nodes', Nodes)
    # print('GroundNodes',  GroundNodes)

    # BC and constraints as matrix
    Cstr_full, Cstr, f = matrix_functions.ConstraintMatrix(NodeData, Nodes, GroundNodes, NN, EI, EJ) 
    return Cstr_full, Cstr, f 


def Constraints_nodes(nodes_tuple: Union[Tuple[NDArray[np.int_], NDArray[np.int_]], Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]], \
                      nodeData_tuple: Union[Tuple[NDArray[np.float_], NDArray[np.float_]], NDArray[np.float_]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constraints_afo_task sets up the constraints on nodes and edges for specific learning task, and for specific step.
    This comes after Setup_constraints which sets them for the whole task

    inputs:
    BigClass: class instance that includes all class instances User_Variables, Network_State, et.c
    problem: str, "measure" for pressure posed on inputs and ground, "dual" for pressure also on outputs and change of resistances  
    """
    InNodes: np.ndarray = nodes_tuple[0]
    GroundNodes: np.ndarray = nodes_tuple[1]
    if len(nodes_tuple)==2:  # system is in measure mode, not dual
        InNodeData = nodeData_tuple
        OutputNodeData: np.ndarray = array([], dtype=int)
        OutputNodes: np.ndarray = array([], dtype=int)
    elif len(nodes_tuple)==3:  # system is in dual mode
        OutputNodeData = nodeData_tuple[1]
        InNodeData = nodeData_tuple[0]
        OutputNodes = nodes_tuple[2]
    NodeData: np.ndarray = np.append(InNodeData, OutputNodeData)
    Nodes: np.ndarray = np.append(InNodes, OutputNodes)
    return NodeData, Nodes, GroundNodes