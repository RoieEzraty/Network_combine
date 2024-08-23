from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros
from typing import TYPE_CHECKING, Callable, Union

import functions


############# Class - User Variables #############


class User_Variables:
    """
    
    """
    def __init__(self, iterations: int, Nin: int, Nout: int, gamma: np.ndarray, \
                 R_update: str, use_p_tag: bool, supress_prints: bool, bc_noise: float) -> None:

        self.iterations: int = iterations
        self.Nin: int = Nin
        self.Nout: int = Nout
        self.NN: int = Nin+Nout+1
        self.gamma: np.ndarray = gamma
        self.use_p_tag: bool = use_p_tag 
        if use_p_tag:
            self.loss_fn: Union[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]] = functions.loss_fn_2samples 
        else:
            self.loss_fn = functions.loss_fn_1sample
        self.R_update: str = R_update  # 'propto' if R=gamma*delta_p
                                       # 'deltaR' if deltaR=gamma*delta_p, gamma should be small
        data_size_each_axis=15  # size of training set is data_size**Nin, don't have to cover all of it
        self.supress_prints = supress_prints
        self.bc_noise = bc_noise

    def create_M(self, M_values: np.ndarray):
      """
      creates the matrix which defines the task, i.e. p_out=M*p_in

      inputs:
      M_values: 1D np.ndarray of all values to be inserted to M, consecutively, regardless of structure

      outputs:
      M: np.ndarray sized [Nout, Nin], matrix defining the task p_out=M*p_in
      """
      self.M: np.ndarray =  M_values[0:self.Nout*self.Nin].reshape(self.Nout, self.Nin)

    def assign_alpha_vec(self, alpha: float) -> None:
      if type(alpha)==float:
        self.alpha_vec = np.tile(alpha, (self.Nout,))
      else:
        print('wrong type for alpha, should be float')
