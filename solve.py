from __future__ import annotations
import numpy as np
import numpy.random as rand
import copy

from numpy.linalg import inv as inv
from typing import Tuple, List
from typing import TYPE_CHECKING

import matrix_functions, plot_functions

if TYPE_CHECKING:
    from Big_Class import Big_Class


############# functions that solve flow #############


def solve_flow_const_K(BigClass: "Big_Class", CstrTuple: Tuple[np.ndarray, np.ndarray, np.ndarray], R_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    solve_flow_const_K solves the flow under given conductance configuration without changing Ks, until simulation converges

    inputs:
    BigClass       - class instance including user variables (Variabs), network structure (Strctr) and networkx (NET) and network state (State) class instances
    u              - 1D array sized [NE + constraints, ], flow field at edges from previous solution iteration
    Cstr           - 2D array without last column, which is f from Rocks & Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
    f              - constraint vector (from Rocks and Katifori 2018)
    iters_same_BSc - # iteration allowed under same boundary conditions (same constraints)

    outputs:
    p     - 1D array sized [NN + constraints, ], pressure at nodes at end of current iteration step
    u_nxt - 1D array sized [NE + constraints, ], flow velocity at edgses at end of current iteration step
    """
    # create effective conductivities if they are flow dependent
    Cstr = CstrTuple[1]
    f = CstrTuple[2]
    K_vec, K_mat = matrix_functions.K_from_R(R_vec, BigClass.Strctr.NE)
    # print('K_vec', K_vec)
    # print('K_mat', K_mat)
    L, L_bar = matrix_functions.buildL(BigClass, BigClass.Strctr.DM, K_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian
    p, u = solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, K_vec, f, round=10**-10)  # pressure and flow
    # plot_functions.PlotNetwork(p, u, K_vec, BigClass, mode='u_p', nodes=True, edges=True, savefig=False)
    # plot_functions.PlotNetwork(p, u, K_vec, BigClass, mode='R_p', nodes=True, edges=True, savefig=False)
    return p, u


# @lru_cache(maxsize=20)
def solve_flow(L_bar: np.ndarray, EI: np.ndarray, EJ: np.ndarray, K: np.ndarray, f: np.ndarray, round: float=10**-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves for the pressure at nodes and flow at edges, given Lagrangian etc.
    flow at edge defined as difference in pressure between input and output nodes time conductivity at each edge
    
    input:
    L_bar  - Full augmented Lagrangian, 2D np.array sized [NNodes + constraints]
    EI, EJ - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    K_vec  - [NE] 2D cubic np.array of conductivities
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


def dot_triple(X, Y, Z):
    return np.dot(X, np.dot(Y, Z))