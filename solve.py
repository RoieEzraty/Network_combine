from __future__ import annotations
import numpy as np

from numpy.linalg import inv as inv
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from typing import TYPE_CHECKING

import matrix_functions

if TYPE_CHECKING:
    from Big_Class import Big_Class


# ==================================
# functions that solve flow
# ==================================


# @lru_cache(maxsize=20)
def solve_flow(BigClass: "Big_Class", CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
               R_vec: NDArray[np.float_], roundto: Optional[float] = 10**-10) -> Tuple[NDArray[np.float_],
                                                                                       NDArray[np.float_]]:
    """
    Solves for the pressure at nodes and flow at edges, given Lagrangian etc.
    flow at edge defined as difference in pressure between input and output nodes time conductivity at each edge.
    2nd part of State.solve_flow_given_problem, Comes after functions.setup_constraints_given_pin.

    input:
    BigClass -       class instance including User_Variables, Network_Structure instances, etc.
    CstrTuple - Tuple consisting - Cstr_full - 2D array without last column, which is f from Rocks & Katifori 2018
                                               https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
                                   Cstr -      Cstr_full without last line
                                   f    -      constraint vector (from Rocks and Katifori 2018)1D np.arrays sized NEdges
                                               such that EI[i] is node connected to EJ[i] at certain edge
    R_vec  - [NE] 2D cubic np.array of resistivities
    round  - float, value below which the absolute value of u and p are rounded to 0.

    output:
    p - hydrostatic pressure, 1D np.array sized NNodes
    u - velocity through each edge 1D np.array sized len(EI)
    """
    Cstr: NDArray[np.float_] = CstrTuple[1]
    f: NDArray[np.float_] = CstrTuple[2]

    # R to K
    K_vec: NDArray[np.float_]  # type hint them
    K_mat: NDArray[np.float_]  # type hint them
    K_vec, K_mat = matrix_functions.K_from_R(R_vec, BigClass.Strctr.NE)  # calculate them

    # Calculate Inverse Lagrangian
    L: NDArray[np.float_]  # type hint them
    L_bar: NDArray[np.float_]  # type hint them
    L, L_bar = matrix_functions.buildL(BigClass, BigClass.Strctr.DM, K_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian

    IL_bar: NDArray[np.float_] = inv(L_bar)

    # pressure p and velocity u
    p: NDArray[np.float_] = np.dot(IL_bar, f)
    u: NDArray[np.float_] = ((p[BigClass.Strctr.EI] - p[BigClass.Strctr.EJ]).T*K_vec)[0]
    p, u = round_small(p, u)
    return p, u


def round_small(p: NDArray[np.float_], u: NDArray[np.float_], roundto: float = 10**-10) -> Tuple[NDArray[np.float_],
                                                                                                 NDArray[np.float_]]:
    """
    round_small rounds values of u and p that are close to 0 to get rid of rounding problems
    """
    p[abs(p) < roundto] = 0  # Correct for very low pressures
    u[abs(u) < roundto] = 0  # Correct for very low velocities
    return p, u


def dot_triple(X, Y, Z):
    return np.dot(X, np.dot(Y, Z))
