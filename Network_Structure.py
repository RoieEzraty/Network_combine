from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros

import matrix_functions, functions
from User_Variables import User_Variables


############# Class - network structure variables #############


class Network_Structure:
    """
    Net_structure class save the structure of the network
    """

    def __init__(self) -> None:
        pass

    def build_incidence(self, Variabs: "User_Variables") -> None:
        """
        build_incidence builds the incidence matrix DM

        inputs:
        Variabs - variables class

        outputs:
        EI         - array, node number on 1st side of all edges
        EJ         - array, node number on 2nd side of all edges
        EIEJ_plots - array, combined EI and EJ, each line is two nodes of edge, for visual ease
        DM         - array, connectivity matrix NE X NN
        NE         - int, # edges in network
        NN         - int, # nodes in network
        """
        self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence(Variabs)