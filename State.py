from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros

import matrix_functions, functions
from Network_Structure import Network_Structure


############# Class - network state variables #############


class State():
    """
    Class with variables that hold information of state of network.
    what ends with _in_t holds all time instances of the variable, each list index is different t
    what ends w/out _in_t is at current time instance self.t
    """
    def __init__(self, supress_prints: str, bc_noise: str, Nin: int, Nout: int) -> None:
        super(State, self).__init__()      
        self.out_in_t: List[np.ndarray] = []
        self.loss_in_t: List[np.ndarray] = []
        self.t: int = 0
        self.out_dual_in_t: List[np.ndarray] = [0.5*np.ones(Nout)]
        self.p_in_t: List[np.ndarray] = [1.0*np.ones(Nin)]
        self.p_drawn_in_t: List[np.ndarray] = []
            
    def initiate_resistances(self, Strctr: "Network_Structure") -> None:
        """
        After using build_incidence, initiate resistances
        """
        self.R_in_t: List[np.ndarray] = [np.ones((Strctr.NE), dtype=float)]
            
    def draw_p(self):
        self.p_drawn = np.random.uniform(low=0.0, high=2.0, size=2)
        self.p_drawn_in_t.append(self.p_drawn)