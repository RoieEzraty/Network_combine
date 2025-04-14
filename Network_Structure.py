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
        elif type == 'beads':
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence_beads(self)
