from __future__ import annotations
import numpy as np

from typing import Tuple, List, Union
from numpy.typing import NDArray
from numpy import array

import matrix_functions


# ===================================================
# other functions
# ===================================================


def loss_fn_2samples(output1: NDArray[np.float_], output2: NDArray[np.float_], desired1: NDArray[np.float_],
                     desired2: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    loss functions for regression task out=M*in using two sampled input pressures

    inputs:
    output1: np.ndarray sized [Nout,] output for 1st sample (current time step)
    output2: np.ndarray sized [Nout,] output for 2nd sample (previous time step)
    desired1: np.ndarray sized [Nout,] desired output for 1st sample (current time step)
    desired2: np.ndarray sized [Nout,] desired output for 2nd sample (previous time step)

    outputs:
    loss: np.ndarray sized [Nout, 2] loss as linear difference output - desired, each line for different sample
    """
    L1: NDArray[np.float_] = desired1-output1
    L2: NDArray[np.float_] = desired2-output2
    loss: NDArray[np.float_] = np.array([L1, L2])
    return loss


def loss_fn_1sample(output: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """
    loss functions for regression task out=M*in using a single drawn input pressure

    inputs:
    output: np.ndarray sized [Nout,] output for 1st sample (current time step)
    desired: np.ndarray sized [Nout,] desired output for 1st sample (current time step)

    outputs:
    loss: np.ndarray sized [Nout, 2] loss as linear difference output - desired, each line for different sample
    """
    L1: NDArray[np.float_] = desired-output
    loss: NDArray[np.float_] = np.array([L1])
    return loss


def setup_constraints_given_pin(nodes_tuple: Union[Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]],
                                                   Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                                         NDArray[np.int_]],
                                                   Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                                         NDArray[np.int_], NDArray[np.int_]],
                                                   Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                                         NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]],
                                nodeData_tuple: Union[Tuple[NDArray[np.float_], NDArray[np.float_]],
                                                      Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
                                                      Tuple[NDArray[np.float_], NDArray[np.float_],
                                                            NDArray[np.float_], NDArray[np.float_]],
                                                      Tuple[NDArray[np.float_], NDArray[np.float_],
                                                            NDArray[np.float_], NDArray[np.float_],
                                                            NDArray[np.float_]]],
                                NN: int,
                                EI: NDArray[np.int_],
                                EJ: NDArray[np.int_]) -> Tuple[NDArray[np.float_], NDArray[np.float_],
                                                               NDArray[np.float_]]:
    """
    setup_constraints_given_pin sets up arrays of boundary condition on nodes,
    denoting node indices and assigned pressure values to each node,
    from which it calculates the constraints matrices Cstr and f.

    inputs:
    nodes_tuple    - Tuple containing indices of nodes: (input_nodes_arr, inter_nodes_arr) for the "measure" problem
                     or (input_nodes_arr, inter_nodes_arr, output_nodes_arr) for the "dual".
    nodeData_tuple - Tuple, pressure values of nodes_tuple: (input_nodes_arr, inter_nodes_arr) for the "measure" problem
                                                            or (input_nodes_arr, inter_nodes_arr, output_nodes_arr)
                                                            for the "dual".
    NN             - int, total number of nodes in network
    EI             - array, node number on 1st side of all edges
    EJ             - array, node number on 2nd side of all edges

    outputs:
    Cstr_full - 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges.
                last column is value of constraint
                (p value of row contains just +1. pressure drop if row contains +1 and -1)
    Cstr      - 2D array without last column
                (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    f         - constraint vector (from Rocks and Katifori 2018)
    """
    # specific constraints for training step
    NodeData: NDArray[np.float_]  # type hint
    Nodes: NDArray[np.int_]  # type hint
    GroundNodes: NDArray[np.int_]  # type hint
    NodeData, Nodes, GroundNodes = Constraints_nodes(nodes_tuple, nodeData_tuple)

    # print('NodeData', NodeData)
    # print('Nodes', Nodes)
    # print('GroundNodes',  GroundNodes)

    # BC and constraints as matrix
    Cstr_full: NDArray[np.float_]  # type hint
    Cstr: NDArray[np.float_]  # type hint
    f: NDArray[np.float_]  # type hint
    Cstr_full, Cstr, f = matrix_functions.ConstraintMatrix(NodeData, Nodes, GroundNodes, NN, EI, EJ)
    return Cstr_full, Cstr, f


def Constraints_nodes(nodes_tuple: Union[Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]],
                                         Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                               NDArray[np.int_]],
                                         Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                               NDArray[np.int_], NDArray[np.int_]],
                                         Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                               NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]],
                      nodeData_tuple: Union[Tuple[NDArray[np.float_], NDArray[np.float_]],
                                            Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
                                            Tuple[NDArray[np.float_], NDArray[np.float_],
                                                  NDArray[np.float_], NDArray[np.float_]],
                                            Tuple[NDArray[np.float_], NDArray[np.float_],
                                                  NDArray[np.float_], NDArray[np.float_],
                                                  NDArray[np.float_]]],) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constraints_nodes sets up the constraints on nodes for specific problem ("measure" or "dual"),
    for specific sampled pressure input_drawn.
    1st part of State.solve_flow_given_problem, after which the flow is solved under solve_flow

    inputs:
    nodes_tuple    - Tuple of indices of nodes: (input_nodes_arr, inter_nodes_arr) for the "measure" problem
                                                or (input_nodes_arr, inter_nodes_arr, output_nodes_arr) for "dual".
    nodeData_tuple - Tuple, pressure values of nodes_tuple: (input_nodes_arr, inter_nodes_arr) for the "measure" problem
                                                            or (input_nodes_arr, inter_nodes_arr, output_nodes_arr)
                                                            for "dual".

    outputs:
    NodeData    - 1D array at length as "Nodes" corresponding to pressures at each node from "Nodes"
    Nodes       - 1D array of nodes that have a constraint
    GroundNodes - 1D array of nodes that have a constraint of ground (outlet)
    """
    InNodes: NDArray[np.int_] = nodes_tuple[0]
    InNodeData: NDArray[np.float_] = nodeData_tuple[0]
    extraInNodes: NDArray[np.int_] = nodes_tuple[1]
    extraInNodeData: NDArray[np.float_] = nodeData_tuple[1]
    GroundNodes: NDArray[np.int_] = nodes_tuple[2]
    if len(nodes_tuple) == 3:  # system is in measure mode, not dual
        OutputNodes: NDArray[np.int_] = array([], dtype=int)
        OutputNodeData: NDArray[np.float_] = array([], dtype=float)
        extraOutputNodes: NDArray[np.int_] = array([], dtype=int)
        extraOutputNodeData: NDArray[np.float_] = array([], dtype=float)
        InterNodes: NDArray[np.int_] = array([], dtype=int)
        InterNodeData: NDArray[np.float_] = array([], dtype=float)
    elif len(nodes_tuple) == 5:  # system is in dual mode
        if len(nodeData_tuple) != 4:
            print('nodeData_tuple incompatible')
        else:
            OutputNodes = nodes_tuple[3]
            OutputNodeData = nodeData_tuple[2]
            extraOutputNodes = nodes_tuple[4]
            extraOutputNodeData = nodeData_tuple[3]
            InterNodes = array([], dtype=int)
            InterNodeData = array([], dtype=float)
    elif len(nodes_tuple) == 6:  # system is in dual mode with inter nodes
        if len(nodeData_tuple) != 5:
            print('nodeData_tuple incompatible')
        else:
            OutputNodes = nodes_tuple[3]
            OutputNodeData = nodeData_tuple[2]
            extraOutputNodes = nodes_tuple[4]
            extraOutputNodeData = nodeData_tuple[3]
            InterNodes = nodes_tuple[5]
            InterNodeData = nodeData_tuple[4]
    # print('InNodeData', InNodeData)
    # print('OutputNodes', OutputNodes)
    # print('OutputNodeData', OutputNodeData)
    # print('InterNodes', InterNodes)
    # print('InterNodeData', InterNodeData)
    NodeData: NDArray[np.float_] = np.append(np.append(np.append(np.append(InNodeData, extraInNodeData), InterNodeData),
                                                       OutputNodeData), extraOutputNodeData)
    Nodes: NDArray[np.int_] = np.append(np.append(np.append(np.append(InNodes, extraInNodes), InterNodes),
                                                  OutputNodes), extraOutputNodes)
    return NodeData, Nodes, GroundNodes
