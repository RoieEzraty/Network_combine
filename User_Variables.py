from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import functions


# ===================================================
# Class - User Variables
# ===================================================


class User_Variables:
    """
    Class with variables given by user.
    These remain the same along the simulation
    """
    def __init__(self, iterations: int, Nin: int, Nout: int, gamma: NDArray[np.float_], R_update: str, use_p_tag: bool,
                 supress_prints: bool, bc_noise: float, Ninter: Optional[int] = 0) -> None:

        self.iterations: int = iterations
        self.Nin: int = Nin
        self.Nout: int = Nout
        self.Ninter: Optional[int] = Ninter if 'Ninter' in locals() else None
        if 'Ninter' in locals() and Ninter is not None:
            self.NN: int = Nin + Nout + Ninter
        else:
            self.NN = Nin + Nout
        self.gamma: NDArray[np.float_] = gamma
        self.use_p_tag: bool = use_p_tag
        if use_p_tag:
            self.loss_fn: Union[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                                Callable[[np.ndarray, np.ndarray], np.ndarray]] = functions.loss_fn_2samples
        else:
            self.loss_fn = functions.loss_fn_1sample
        self.R_update: str = R_update  # 'propto' if R=gamma*delta_p
                                       # 'deltaR' if deltaR=gamma*delta_p, gamma should be small
        data_size_each_axis = 15  # size of training set is data_size**Nin, don't have to cover all of it
        self.supress_prints = supress_prints
        self.bc_noise = bc_noise

    def create_M(self, M_values: NDArray[np.float_]) -> None:
        """
        creates the matrix which defines the task, i.e. p_out=M*p_in

        inputs:
        M_values: 1D np.ndarray of all values to be inserted to M, consecutively, regardless of structure

        outputs:
        M: np.ndarray sized [Nout, Nin], matrix defining the task p_out=M*p_in
        """
        self.M: np.ndarray = M_values[0:self.Nout*self.Nin].reshape(self.Nout, self.Nin)

    def assign_alpha_vec(self, alpha: float) -> None:
        """
        assign the alpha vector, in the form of array of [Nout], to the User_Variables

        inputs:
        alpha: float of the learning rate alpha
        """
        if isinstance(alpha, float):
            self.alpha_vec: NDArray[np.float_] = np.tile(alpha, (self.Nout,))
        else:
            print('wrong type for alpha, should be float')
