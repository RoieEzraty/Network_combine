from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy.typing import NDArray
from numpy import array, zeros
from typing import TYPE_CHECKING

import matrix_functions, functions


if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Big_Class import Big_Class


# ===================================================
# Class - network structure variables
# ===================================================


class Network_Structure:
    """
    Net_structure class save the structure of the network
    """

    def __init__(self, input_nodes_arr: NDArray[np.int_], output_nodes_arr: NDArray[np.int_],
                 inter_nodes_arr: NDArray[np.int_], ground_nodes_arr: NDArray[np.int_]) -> None:
        self.input_nodes_arr: NDArray[np.int_] = input_nodes_arr
        self.output_nodes_arr: NDArray[np.int_] = output_nodes_arr
        self.inter_nodes_arr: NDArray[np.int_] = inter_nodes_arr
        self.ground_nodes_arr: NDArray[np.int_] = ground_nodes_arr

    def build_incidence(self) -> None:
        """
        build_incidence builds the incidence matrix DM

        inputs:
        None

        outputs:
        EI         - array, node number on 1st side of all edges
        EJ         - array, node number on 2nd side of all edges
        EIEJ_plots - array, combined EI and EJ, each line is two nodes of edge, for visual ease
        DM         - array, connectivity matrix NE X NN
        NE         - int, # edges in network
        NN         - int, # nodes in network
        """
        self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence(self)

    def build_edges(self) -> None:
        """
        assign arrays denoting edges of the network to the Network_Structure instance using the EI and EJ
        """
        self.input_edges: NDArray[np.int_]  # type hint
        self.inter_edges: NDArray[np.int_]  # type hint
        self.output_edges: NDArray[np.int_]  # type hint
        self.ground_edges: NDArray[np.int_]  # type hint
        self.input_edges = matrix_functions.edges_from_EI_EJ(self.input_nodes_arr, self.EI, self.EJ)
        self.inter_edges = matrix_functions.edges_from_EI_EJ(self.inter_nodes_arr, self.EI, self.EJ)
        self.output_edges = matrix_functions.edges_from_EI_EJ(self.output_nodes_arr, self.EI, self.EJ)
        self.ground_edges = matrix_functions.edges_from_EI_EJ(self.ground_nodes_arr, self.EI, self.EJ)
