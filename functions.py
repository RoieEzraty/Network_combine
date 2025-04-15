from __future__ import annotations
import numpy as np

from typing import Tuple, List, Union, Optional
from numpy.typing import NDArray
from numpy import array
from typing import TYPE_CHECKING, Literal
from typing_extensions import Annotated

import matrix_functions


# ===================================================
# other functions
# ===================================================


def loss_fn_2samples(output1: NDArray[np.float_], output2: NDArray[np.float_],
                     desired1: NDArray[np.float_], desired2: NDArray[np.float_],
                     Power1: Optional[np.float_] = None, Power2: Optional[np.float_] = None,
                     lam: Optional[np.float_] = None) -> NDArray[np.float_]:
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
    # L1: NDArray[np.float_] = np.abs(desired1-output1)
    # L2: NDArray[np.float_] = np.abs(desired2-output2)
    L1: NDArray[np.float_] = desired1-output1
    L2: NDArray[np.float_] = desired2-output2
    loss: NDArray[np.float_] = np.array([L1, L2])

    if Power1:  # add Power to loss, as in Stern 2024 power-efficiency arXiv:2310.10437v1
        if not Power2 or not lam:
            print('not enough arguments input to loss function')
        else:
            print('loss before power', loss)
            print('delta_loss', lam * np.array([[Power1], [Power2]]))
            loss += lam * np.array([[Power1], [Power2]])

    return loss


def loss_fn_1sample(output: np.ndarray, desired: np.ndarray,
                    Power: Optional[np.float_] = None,
                    lam: Optional[np.float_] = None) -> np.ndarray:
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

    if Power:  # add Power to loss, as in Stern 2024 power-efficiency arXiv:2310.10437v1
        if not lam:
            print('not enough arguments input to loss function')
        else:
            print('using power')
            print('loss before power', loss)
            loss += lam * Power
            print('loss after power ', loss)

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
    nodes_tuple    - Tuple containing indices of nodes: (input_nodes_arr, inter_nodes_arr) for "measurement" modality
                     or (input_nodes_arr, inter_nodes_arr, output_nodes_arr) for the "dual".
    nodeData_tuple - Tuple, pressure values of nodes_tuple: (input_nodes_arr, inter_nodes_arr) for "measurement" modality
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
    Assemble node constraints and their pressure values for solving flow in a network.

    Parameters
    ----------
    nodes_tuple : tuple of np.ndarray
        Indices of constrained nodes.
        Supported formats:
        - (in_nodes, extra_in_nodes, ground_nodes)
        - + extra_output_nodes
        - + output_nodes, extra_output_nodes
        - + inter_nodes
    nodeData_tuple : tuple of np.ndarray
        Corresponding pressure values for the constrained nodes.

    Returns
    -------
    NodeData : np.ndarray
        Pressure values of all constrained nodes, in order.
    Nodes : np.ndarray
        Indices of constrained nodes, in the same order as NodeData.
    GroundNodes : np.ndarray
        Indices of ground nodes (outlets) for pressure reference.
    """
    # Required core inputs
    InNodes, extraInNodes, GroundNodes = nodes_tuple[:3]
    InNodeData, extraInNodeData = nodeData_tuple[:2]

    # Initialize optional components
    OutputNodes = np.array([], dtype=int)
    OutputNodeData = np.array([], dtype=float)
    extraOutputNodes = np.array([], dtype=int)
    extraOutputNodeData = np.array([], dtype=float)
    InterNodes = np.array([], dtype=int)
    InterNodeData = np.array([], dtype=float)

    # Dispatch based on additional inputs
    if len(nodes_tuple) >= 4:
        extraOutputNodes = nodes_tuple[3]
        extraOutputNodeData = nodeData_tuple[2]
    if len(nodes_tuple) >= 5:
        OutputNodes = nodes_tuple[3]
        extraOutputNodes = nodes_tuple[4]
        OutputNodeData = nodeData_tuple[2]
        extraOutputNodeData = nodeData_tuple[3]
    if len(nodes_tuple) == 6:
        InterNodes = nodes_tuple[5]
        InterNodeData = nodeData_tuple[4]

    # Combine node data and indices in correct order
    NodeData = np.concatenate([InNodeData, extraInNodeData, InterNodeData, OutputNodeData, extraOutputNodeData])
    Nodes = np.concatenate([InNodes, extraInNodes, InterNodes, OutputNodes, extraOutputNodes])

    return NodeData, Nodes, GroundNodes


def random_gen_M(random_state: int, size: int) -> NDArray[np.float_]:
    """
    random_gen_M generates a random M_values array for regression task
    use for multiple_Nin_Nout for example, and before train_loop()

    inputs:
    random_state - int, random seed
    size         - int, size of M_values, train_loop then decides how many to take

    output:
    1D [Nin*Nout] array of random values for task matrix M
    """
    # generate random state
    random_gen = np.random.RandomState(random_state)

    # Generate random values with the defined random state
    M_values = random_gen.rand(size)

    return M_values


def normalize_M(M_values: NDArray[np.float_],
                normalization: float,
                Nin: int,
                Nout: int
                ) -> Annotated[NDArray[np.float_], "shape: (Nin*Nout,)"]:
    """
    normalize_M creates normalized task matrix M given un-normalized values of M

    inputs:
    M_values      - 1D NDarray of un-normalized values of task matrix
    normalization - the all values in every line M sum up to normalization
    Nin           - int, # inputs
    Nout          - int, # outputs

    output:
    M_values_norm - 1D NDArray [Nin*Nout] of values for task matrix,
                    normalized so each line in M matrix sums up to normalization
    """
    # generate random state
    M_mat: Annotated[NDArray[np.float_], "shape: (Nin, Nout)"] = M_values[0:Nout*Nin].reshape(Nout, Nin)
    M_line: NDArray[np.float_] = np.sum(M_mat, axis=1)
    M_values_norm = M_values[:Nin*Nout]/np.max(M_line)*normalization  # max sum over line = "normalization"
    return M_values_norm


def reset_update(vec: NDArray[np.float_], reset_thresh_b: float, reset_thresh_s: float) -> bool:
    """
    Reset the input_update_nxt vector if any values diverge too far from expected bounds.

    Parameters
    ----------
    vec : NDArray[np.float_]
        update modality vector (e.g. State.input_update_nxt)
    reset_thresh_b : float
        Threshold for detecting abnormally large input values (above this).
    reset_thresh_s : float
        Threshold for detecting abnormally small input values (below this).

    Returns
    -------
    boolean whether to reset update values or not
    """
    # find indices in input_update_nxt where values diverge
    reset_inds_big = vec > reset_thresh_b
    reset_inds_small = vec < reset_thresh_s
    # reset them to initial value
    # self.input_update_nxt[reset_inds] = self.input_update_in_t[0][reset_inds]
    if np.any(array([reset_inds_big, reset_inds_small])):
        return True
    else:
        return False
