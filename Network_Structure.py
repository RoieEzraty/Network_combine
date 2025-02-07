from __future__ import annotations
import numpy as np

from typing import Tuple
from numpy.typing import NDArray

import matrix_functions


# ===================================================
# Class - network structure variables
# ===================================================


class Network_Structure:
    """
    Net_structure class save the structure of the network
    """

    def __init__(self, inOutInterGround_tuple: Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                                     NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],],
                 net_type: str = 'FC', height: int = 0, length: int = 0) -> None:
        """
        net_types:
        FC                   - each input connected to each output
        FC_connected_outputs - FC and all outputs connected
        partialInter         - each input connected to an inter node, that inter node to an output node
        square               - N*N array of nodes, each node has 4 neighbors, some are inputs and some outputs
        """
        self.input_nodes_arr: NDArray[np.int_] = inOutInterGround_tuple[0]
        self.extraInput_nodes_arr: NDArray[np.int_] = inOutInterGround_tuple[1]
        self.inter_nodes_arr: NDArray[np.int_] = inOutInterGround_tuple[2]
        self.output_nodes_arr: NDArray[np.int_] = inOutInterGround_tuple[3]
        self.extraOutput_nodes_arr: NDArray[np.int_] = inOutInterGround_tuple[4]
        self.ground_nodes_arr: NDArray[np.int_] = inOutInterGround_tuple[5]

        # for square network
        self.net_type = net_type
        self.net_height = height
        self.net_len = length

    def build_incidence(self, type: str = 'FC') -> None:
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
        if type == 'FC' or type == 'FC_connected_outputs':
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence(self)
        elif type == 'partialInter':
            print('partialInter is true')
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN =\
                matrix_functions.build_incidence_partialInter(self)
        elif type == 'square':
            print('building square network')
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence_square(self)

    def build_edges(self) -> None:
        """
        assign arrays denoting edges of the network to the Network_Structure instance using the EI and EJ
        """
        self.input_edges: NDArray[np.int_]  # type hint
        self.extraInput_edges: NDArray[np.int_]  # type hint
        self.inter_edges: NDArray[np.int_]  # type hint
        self.output_edges: NDArray[np.int_]  # type hint
        self.extraOutput_edges: NDArray[np.int_]  # type hint
        self.ground_edges: NDArray[np.int_]  # type hint
        self.input_edge_directions: NDArray[np.int_]  # type hint
        self.extraInput_edge_directions: NDArray[np.int_]  # type hint
        self.inter_edge_directions: NDArray[np.int_]  # type hint
        self.output_edge_directions: NDArray[np.int_]  # type hint
        self.extraOutput_edge_directions: NDArray[np.int_]  # type hint
        self.ground_edge_directions: NDArray[np.int_]  # type hint
        self.input_edges = matrix_functions.edges_from_EI_EJ(self.input_nodes_arr, self.EI, self.EJ)
        self.extraInput_edges = matrix_functions.edges_from_EI_EJ(self.extraInput_nodes_arr, self.EI, self.EJ)
        self.inter_edges = matrix_functions.edges_from_EI_EJ(self.inter_nodes_arr, self.EI, self.EJ)
        self.output_edges = matrix_functions.edges_from_EI_EJ(self.output_nodes_arr, self.EI, self.EJ)
        self.extraOutput_edges = matrix_functions.edges_from_EI_EJ(self.extraOutput_nodes_arr, self.EI, self.EJ)
        self.ground_edges = matrix_functions.edges_from_EI_EJ(self.ground_nodes_arr, self.EI, self.EJ)
        self.input_edge_directions = \
            matrix_functions.edge_directions_from_EI(self.input_nodes_arr, self.EI, self.input_edges)
        self.extraInput_edge_directions = \
            matrix_functions.edge_directions_from_EI(self.extraInput_nodes_arr, self.EI, self.extraInput_edges)
        self.inter_edge_directions = \
            matrix_functions.edge_directions_from_EI(self.inter_nodes_arr, self.EI, self.inter_edges)
        self.output_edge_directions = \
            matrix_functions.edge_directions_from_EI(self.output_nodes_arr, self.EI, self.output_edges)
        self.extraOutput_edge_directions =\
            matrix_functions.edge_directions_from_EI(self.extraOutput_nodes_arr, self.EI, self.extraOutput_edges)
        self.ground_edge_directions = \
            matrix_functions.edge_directions_from_EI(self.ground_nodes_arr, self.EI, self.ground_edges)
