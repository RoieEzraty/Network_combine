from __future__ import annotations
import numpy as np

from typing import Tuple, List
from numpy import array, zeros
from numpy.linalg import norm, inv
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Big_Class import Big_Class


# ==================================
# functions for statistical analysis
# ==================================


def final_err(BigClass: "Big_Class", samples: int = 40):
    """
    Final relative error over the last `samples` time steps.

    This function evaluates the mean absolute error of the loss values from the
    simulation over the last `samples` time steps and normalizes it by the mean
    absolute value of the target output in that same period.

    inputs:
    BigClass - Class instance containing User_Variables, Network_Structure, etc.
    samples  - int, optional Number of most recent time steps to include in the error calculation

    output:
    float of final relative error: mean absolute loss over the last `samples` time steps,
        normalized by the mean absolute value of the corresponding target values.
    """
    numerator = np.mean(np.mean(np.mean(np.abs(BigClass.State.loss_in_t), axis=1), axis=1)[-samples:])
    denominator = np.mean(np.abs(BigClass.Variabs.targets)[-samples:])
    return numerator / denominator


def calculate_accuracy_1sample(output, targets_mat: NDArray[np.float_], target_i: NDArray[np.float_]) -> float:
    """
    Classification accuracy for a single iris sample using L2 distance.

    Compares network output vector to a matrix of possible class targets targets_mat
    and selects the class whose target vector is closest in L2 norm. It then compares
    the predicted class to the true class target_i (provided as a tokenized vector).

    Parameters
    ----------
    output : np.ndarray
        Output vector from the network for a single sample.
    targets_mat : np.ndarray of shape (3, 3)
        A matrix containing the prototype vectors for each class.
    target_i : np.ndarray of shape (3,)
        The one-hot encoded true target class for this sample.

    Returns
    -------
    float
        1.0 if the predicted class matches the true class, otherwise 0.0.
    """
    l2_vec = np.zeros(3)
    for i in range(3):
        l2_vec[i] = norm(output - targets_mat[i])

    # problematic line, wrapped in try-except
    try:
        accuracy: int = int(np.where(l2_vec == np.min(l2_vec)) == np.where(target_i == 1.))
    except ValueError as e:
        # Print the values to inspect what's happening
        print(f"output: {output}")
        print(f"targets_mat: {targets_mat}")
        print(f"l2_vec: {l2_vec}")
        print(f"np.min(l2_vec): {np.min(l2_vec)}")
        print(f"np.where(l2_vec == np.min(l2_vec)): {np.where(l2_vec == np.min(l2_vec))}")
        print(f"np.where(target_ind == 1.): {np.where(target_i == 1.)}")
        print(f"ValueError: {e}")
        accuracy = 0  # or handle it in a way that's meaningful to your logic

    return accuracy


def power_dissip(u: NDArray[np.float_], R: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Calculates the power dissipation in network given flow and resistances on edges

    input:
    u - 1D np.array [Nedges] of velocities at edges
    R - 1D np.array [Nedges] of resistances at edges

    output:
    P - float, power dissipation in network
    """
    P = np.sum(u**2 * R)
    return P


def power_dissip_norm(u: NDArray[np.float_], R: NDArray[np.float_], input: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Calculates the power dissipation in network, normalized by the input pressure squared, given flow and resistances

    input:
    u     - 1D np.array [Nedges] of velocities at edges
    R     - 1D np.array [Nedges] of resistances at edges
    input - 1D np.array [Nin] of input pressure

    output:
    P_norm - float, power dissipation in network
    """
    input_squared = np.mean(input**2)
    P_norm = np.sum(u**2 * R)/input_squared
    return P_norm


def mov_ave(data: NDArray[np.float_], window_size: int) -> NDArray[np.float_]:
    """
    Apply a simple moving average filter to 1D data over a specified window size
    using convolution.

    Parameters
    ----------
    data : np.ndarray
        1D input data to be smoothed.
    window_size : int
        The number of elements to average over.

    Returns
    -------
    np.ndarray
        The smoothed data array after applying the moving average.
        Its length is `len(data) - window_size + 1`.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors a and b.

    Parameters
    ----------
    a,b : np.ndarrays

    Returns
    -------
    float
        Cosine similarity between vectors `a` and `b`, ranging [-1, 1]
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def dK(R_in_t: list[NDArray[np.float_]]) -> NDArray[np.float_]:
    """
    change in conductances during latest time step, calculated by difference between two inverse resistance vectors

    input:
    R_in_t - List of 1D np.arrays [NE] of edge resistance values in time (list rows)

    output:
    dK - 1D np.array [NE] change in conductivity between time steps
    """
    return 1/R_in_t[-1]-1/R_in_t[-2]


# # NOT IN USE


# def dw_Balasub(alpha: float, p: NDArray[np.float_], in_nodes: NDArray[np.int_], out_nodes: NDArray[np.int_],
#                DM: NDArray[np.int_], R: NDArray[np.float_], M: NDArray[np.float_]) -> NDArray[np.float_]:
#     """
#     dw_Balasub calculates the change in weights (conductances) by contrastive learning
#     from the analytic derivation in Stern & Balasubramanian 2024 https://doi.org/10.1103/PhysRevE.109.024311
#     (equation 17)

#     input:
#     alpha     - float, learning rate
#     p         - 1D np.array [NN] of pressures on all network nodes
#     out_nodes - 1D np.array [Nout] indices of output nodes from the pressure vector p
#     DM        - np.array [NEdges, NNodes], incidence matrix
#     R         - 1D np.array [NE] resistances of edges
#     M         - np.array [Nout, Nin] regression task matrix

#     output:
#     dw - 1D np.array [NE] change in conductivities (inverse of resistances) as in contrastive learning
#     """
#     epsilon = 10**-5  # add for positive definiteness
#     Nin = np.size(in_nodes)  # number of input nodes
#     NN = np.size(p)  # number of nodes in network

#     diagw = np.diag(1/R)  # diagonal [NN, NN] of conductivities
#     H = DM.T@diagw@DM  # Hessian
#     eps_diag = np.diag(np.ones(NN, dtype=np.float64)) * epsilon
#     H = H + eps_diag  # add for positive definiteness
#     invH = inv(H)  # invert H
#     A = np.zeros([NN, 1])  # desired response vector [NN]
#     A[out_nodes] = M @ np.ones([Nin, 1])  # desired response is only at outputs
#     B = np.sum(p[out_nodes])  # desired response scalar
#     dw = -alpha*B*(DM@p)*(DM@invH@A)  # change in conductivities
#     return dw


# def flow_MSE(u: NDArray[np.float_], step: int, u_nxt=[]) -> NDArray[float_]:
# 	"""
# 	flow_MSE calculates the MSE between different instances of u given iteration difference step

# 	input:
# 	u     - if u_nxt == []: 2D np.array [NEdges, iteration] of velocities at each edge (rows) and iteration step (cols)
# 			else:           1D np.array [Nedges] of velocities at early simulation step
# 	step  - calculate the MSE at step steps
# 	u_nxt - optional 1D np.array [Nedges] of velocities at later simulation step

# 	output:
# 	MSE - 1D np.array [iteration/step] of MSE between velocities at different simulation steps
# 	"""

# 	if len(u_nxt) == 0:
# 		MSE = np.zeros([np.shape(u)[1],])
# 		for i in range(np.shape(u)[1]-step):
# 			MSE[i] = np.sqrt(np.square(u[:, i+step] - u[:, i]).mean())
# 	else:
# 		MSE = np.sqrt(np.square(u_nxt - u).mean())
# 	return MSE


# def K_Hamming(K_cells, step, K_cells_nxt=[]):
# 	"""
# 	K_Hamming calculates the Hamming between different conductivity constellations given iteration difference step

# 	input:
# 	K_cells       - if K_cells_nxt == []: 2D np.array [NGrids**2, iteration] of edge conductivities at each edge (rows) and iteration step (cols)
# 			        else:                 1D np.array [NGrids**2] of conductivities at early simulation step
# 	step          - calculate the Hamming distance at step steps
# 	K_cells_nxt   - optional 1D np.array [Nedges] of conductivities at later simulation step

# 	output:
# 	Hamming - 1D np.array [iteration/step] of Hamming distance between edge conductivity constellations at different simulation steps
# 	"""

# 	if len(K_cells_nxt) == 0:
# 		Hamming = np.zeros([np.shape(K_cells)[1],])
# 		for i in range(np.shape(K_cells)[1]-step):
# 			Hamming[i] = np.mean(K_cells[:, i+step] != K_cells[:, i])
# 	else:
# 		Hamming = np.mean(K_cells_nxt != K_cells)
# 	return Hamming


# def calc_ratio_loss(output, target, input_p):
# 	"""
# 	calc_ratio calculates the ratio between output and targets

# 	inputs:
# 	output   - array, output of flow through ground nodes
# 	target   - array, desired target output
# 	input_p  - array for Regression, float for rest (I think), pressure at input nodes

# 	output:
# 	ratio - float
# 	"""
# 	if np.size(output)>1:
# 		# ratio_loss = (target[0]/target[1]-output[0]/output[1])/(target[0]/target[1] - 1)
# 		ratio_loss = np.mean(np.abs((target - output)/(target)))
# 	else:
# 		# p_noBalls = np.sum(input_p)/(2+.64)  # numerically calculated, theoretical p at output as if no balls, large nets.
# 		# ratio_loss = (target - output)/(target - p_noBalls)
# 		ratio_loss = np.abs((target - output)/(target))
# 		R_tot = np.sum(input_p)/output
# 		print('R tot ', R_tot)
# 	return ratio_loss


# def calculate_p_nudge(BigClass, State, error=0.0, p_in=0.0, error_prev=0.0, p_in_prev=0.0):
# 	"""
# 	calculate_p_nudge calculates the nudged pressure - whether in contrastive learning or update modality

# 	inputs:
# 	BigClass  - class instance with all relevant data
# 	p_desired - 1d array of desired outputs at output nodes
# 	error     - 1d array of error from desired measure for the dual case, sized [2,]

# 	outputs:
# 	p_nudge - 1d array of pressure values to be assigned to output nodes, for the clamped stage
# 	"""
# 	if BigClass.Variabs.flow_scheme=='dual':
# 		print('p_nudge', State.p_nudge)
# 		print('error', error)
# 		if BigClass.Variabs.task_type=='Allostery_contrastive':
# 			p_nudge = State.p_nudge - np.dot(BigClass.Variabs.alpha, error)
# 			outputs_dual = State.outputs_dual + BigClass.Variabs.alpha * error
# 		elif BigClass.Variabs.task_type=='Regression_contrastive':
# 			# p_nudge = State.p_nudge - BigClass.Variabs.alpha * error
# 			print('p_in', p_in)
# 			print('error_prev', error_prev)
# 			print('p_in_prev', p_in_prev)
# 			p_nudge = State.p_nudge - BigClass.Variabs.alpha * (p_in - p_in_prev) * (error - error_prev)
# 			print('State.outputs_dual', State.outputs_dual)
# 			outputs_dual = State.outputs_dual + BigClass.Variabs.alpha * (error - error_prev)
# 			print('outputs_dual next', outputs_dual)
# 		return p_nudge, outputs_dual
# 	else:
# 		p_nudge = BigClass.Variabs.etta*BigClass.Variabs.p_desired + (1-BigClass.Variabs.etta)*State.p_outputs
# 		return p_nudge