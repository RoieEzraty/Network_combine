from __future__ import annotations
import numpy as np
import numpy.random as rand
import copy

from numpy.linalg import inv as inv
from typing import Tuple, List


############# functions that solve flow #############


# @lru_cache(maxsize=20)
def Solve_flow(L_bar: np.ndarray, EI: np.ndarray, EJ: np.ndarray, K: np.ndarray, f: np.ndarray, round: float=10**-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves for the pressure at nodes and flow at edges, given Lagrangian etc.
    flow at edge defined as difference in pressure between input and output nodes time conductivity at each edge
    
    input:
    L_bar  - Full augmented Lagrangian, 2D np.array sized [NNodes + constraints]
    EI, EJ - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    K      - [NE] 2D cubic np.array of conductivities
    f      - constraint vector (from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
    round  - float, value below which the absolute value of u and p are rounded to 0. 
    
    output:
    p - hydrostatic pressure, 1D np.array sized NNodes 
    u - velocity through each edge 1D np.array sized len(EI) 
    """
    
    # Inverse Lagrangian
    IL_bar = inv(L_bar)
    
    # pressure p and velocity u
    p = np.dot(IL_bar,f)
    u = ((p[EI] - p[EJ]).T*K)[0]

    p, u = round_small(p, u)
    return p, u


def round_small(p, u):
    """
    round_small rounds values of u and p that are close to 0 to get rid of rounding problems
    """

    p[abs(p)<10**-10] = 0  # Correct for very low pressures
    u[abs(u)<10**-10] = 0  # Correct for very low velocities

    return p, u