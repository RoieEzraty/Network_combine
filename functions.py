from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros


############# other functions #############


def loss_fn_regression(output1: np.ndarray, output2: np.ndarray, desired1: np.ndarray, desired2: np.ndarray) -> np.ndarray:
    L1 = desired1-output1
    L2 = desired2-output2
    loss = np.array([L1, L2])
    return loss