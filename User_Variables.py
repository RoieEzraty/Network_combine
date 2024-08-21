from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros

import functions


############# Class - User Variables #############


class User_Variables:
    """
    
    """
    def __init__(self, iterations: int, input_nodes_lst: np.ndarray, output_nodes: np.ndarray, \
                 ground_nodes_lst: np.ndarray, Nin: int, Nout: int, alpha_vec: np.ndarray, gamma: np.ndarray, \
                 task_type: str, R_update: str, use_p_tag: bool, supress_prints: bool, bc_noise: float) -> None:

        self.iterations = iterations
        self.Nin = Nin
        self.Nout = Nout
        self.gamma = gamma
        self.use_p_tag = use_p_tag 
        self.R_update = R_update  # 'propto' if R=gamma*delta_p
                                  # 'deltaR' if deltaR=gamma*delta_p, gamma should be small
        data_size_each_axis=15  # size of training set is data_size**Nin, don't have to cover all of it
        # self.train_data, self.train_target, self.test_data, self.test_target = Matrixfuncs.create_regression_dataset(data_size_each_axis, Nin, desired_p_frac, train_frac)
        
        # initalized drawn sample vec and loss func
        self.loss_fn = functions.loss_fn_regression
        self.p_drawn_in_t: List[np.ndarray] = []
        self.desired_in_t: List[np.ndarray] = []
        self.supress_prints = supress_prints
        self.bc_noise = bc_noise
        self.input_nodes_lst = input_nodes_lst
        self.output_nodes = output_nodes
        self.ground_nodes_lst = ground_nodes_lst