from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import functions, solve

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Big_Class import Big_Class


# ===================================================
# Class - network state variables
# ===================================================


class Network_State:
    """
    Class with variables that hold information of state of network.
    what ends with _in_t holds all time instances of the variable, each list index is different t
    what ends w/out _in_t is at current time instance self.t
    """
    def __init__(self, Nin: int, Nout: int, Ninter: Optional[int] = None) -> None:
        super().__init__()
        self.t: int = 0  # time, defined as number of R updates, i.e. times the learning rate alpha is used.
        self.p: NDArray[np.float_] = array([])  # pressure
        self.u: NDArray[np.float_] = array([])  # flow rate
        self.input_drawn_in_t: List[NDArray[np.float_]] = []  # pressure at inputs in time, sampled
        if Ninter is not None:
            self.inter_in_t: List[NDArray[np.float_]] = []
        self.output_in_t: List[NDArray[np.float_]] = []  # pressure at outputs in time
        self.desired_in_t: List[NDArray[np.float_]] = []
        self.input_dual_in_t: List[NDArray[np.float_]] = [1.0 * np.ones(Nin)]
        if Ninter is not None:
            self.inter_dual_in_t: List[NDArray[np.float_]] = [np.random.random(Ninter)]
        self.output_dual_in_t: List[NDArray[np.float_]] = [0.5 * np.ones(Nout)]
        self.loss_in_t: List[NDArray[np.float_]] = []

    def initiate_resistances(self, BigClass: "Big_Class", R_vec_i: Optional[NDArray[np.float_]] = None) -> None:
        """
        After using build_incidence, initiate resistances

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        R_vec_i  - initial resistances, array of size [NE,]
        """
        if R_vec_i is not None:
            self.R_in_t: List[NDArray[np.float_]] = [R_vec_i]
        else:
            self.R_in_t = [np.ones((BigClass.Strctr.NE), dtype=float)]

    def draw_p_in_and_desired(self, Variabs: "User_Variables", i: int) -> None:
        """
        Every time step, draw random input pressures and calculate the desired output given input

        inputs:
        Variabs - User_Variables class
        i       - int, iteration #

        outputs
        input_drawn: np.ndarray sized [Nin,], input pressures
        desired: np.ndarray sized [Nout,], desired output defined by the task M*p_input
        """
        self.input_drawn: NDArray[np.float_] = Variabs.dataset[i % np.shape(Variabs.dataset)[0]]
        if Variabs.task_type == 'Iris_classification':
            self.desired: NDArray[np.float_] = np.matmul(Variabs.targets[i % np.shape(Variabs.dataset)[0]],
                                                         self.target_mat)
            print('onehot_target', Variabs.targets[i % np.shape(Variabs.dataset)[0]])
        else:
            self.desired = Variabs.targets[i % np.shape(Variabs.dataset)[0]]
        self.input_drawn_in_t.append(self.input_drawn)
        self.desired_in_t.append(self.desired)
        if Variabs.supress_prints:
            pass
        else:  # print
            print('input_drawn', self.input_drawn)
            print('desired output=', self.desired)

    def draw_p_means_Iris(self, Variabs: "User_Variables", i: int) -> None:
        """
        Draw input pressure as mean value of every class of Iris dataset.

        inputs:
        Variabs - User_Variables class
        i       - int, iteration # rangin {0-2}

        outputs
        input_drawn: np.ndarray sized [Nin,], input pressures
        """
        self.input_drawn = Variabs.means[i]

    def assign_targets_Iris(self, targets_mat):
        """
        """
        self.target_mat = targets_mat

    def solve_flow_given_problem(self, BigClass: "Big_Class", problem: str) -> None:
        """
        Calculates the constraint matrix Cstr, then solves the flow,
        using functions from functions.py and solve.py,
        given the problem in problem variable.

        inputs:
        BigClass  - class instance including User_Variables, Network_Structure instances, etc.
        problem   - string stating the problem type: "measure" for no constraint on outputs
                                                     "measure_for_mean" for outputs of mean of Iris class
                                                     "dual" for constrained outputs as well

        outputs:
        p - pressure at every node under the specific BC, after convergence while allowing conductivities to change
        u - flow at every edge under the specific BC, after convergence while allowing conductivities to change
        """
        # Calculate pressure p and flow u
        if problem == 'measure' or problem == 'measure_for_mean':
            CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]  # type hint
            CstrTuple = functions.setup_constraints_given_pin(
                        (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.ground_nodes_arr),
                        self.input_drawn, BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
        elif problem == 'dual':
            if BigClass.Variabs.access_interNodes:  # if dual problem accesses interNodes separately
                CstrTuple = functions.setup_constraints_given_pin(
                            (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.ground_nodes_arr,
                             BigClass.Strctr.output_nodes_arr, BigClass.Strctr.inter_nodes_arr),
                            (self.input_dual_in_t[-1], self.output_dual_in_t[-1], self.inter_dual_in_t[-1]),
                            BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
            else:  # if dual problem does not access interNodes separately
                CstrTuple = functions.setup_constraints_given_pin(
                            (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.ground_nodes_arr,
                             BigClass.Strctr.output_nodes_arr),
                            (self.input_dual_in_t[-1], self.output_dual_in_t[-1]),
                            BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
        self.p, self.u = solve.solve_flow(BigClass, CstrTuple, self.R_in_t[-1])

        # Update the State class variables
        if problem in {'measure', 'measure_for_mean'}:
            # Output is at output nodes, ravel so sizes [Nout,]
            self.output: NDArray[np.float_] = self.p[BigClass.Strctr.output_nodes_arr].ravel()
            if BigClass.Variabs.supress_prints:
                pass
            else:  # print
                print('output measured=', self.output)

            if problem == 'measure':  # Only save in time if measuring during training
                self.output_in_t.append(self.output)
                if BigClass.Variabs.access_interNodes:
                    self.inter_in_t.append(self.p[BigClass.Strctr.inter_nodes_arr].ravel())

        # print('Rs', self.R_in_t[-1])

    def calc_loss(self, BigClass: "Big_Class") -> None:
        """
        Calculates the loss given system state and desired outputs, perhaps including 1 time step ago

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        loss: np.ndarray sized [Nout,]
        """
        if BigClass.Variabs.loss_fn == functions.loss_fn_2samples:
            self.loss: NDArray[np.float_] = BigClass.Variabs.loss_fn(self.output, self.output_in_t[-2], self.desired,
                                                                     self.desired_in_t[-2])
        elif BigClass.Variabs.loss_fn == functions.loss_fn_1sample:
            self.loss = BigClass.Variabs.loss_fn(self.output, self.desired)
        self.loss_in_t.append(self.loss)

    def update_input_dual(self, BigClass: "Big_Class") -> None:
        """
        Calculates next input pressure values in dual problem given the measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        input_dual_nxt: np.ndarray sized [Nin,] denoting input pressure of dual problem at time t
        """
        self.t += 1  # update time
        loss: NDArray[np.float_] = self.loss_in_t[-1]  # copy loss
        input_dual: NDArray[np.float_] = self.input_dual_in_t[-1]
        input_drawn: NDArray[np.float_] = self.input_drawn_in_t[-1]
        # dot product for alpha in pressure update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            input_drawn_prev: NDArray[np.float_] = self.input_drawn_in_t[-2]
            delta: NDArray[np.float_] = (input_drawn-input_drawn_prev)*np.dot(BigClass.Variabs.alpha_vec,
                                                                              loss[0]-loss[1])
        else:  # if one sample of p in for every loss calcaultion are to be taken
            delta = (input_drawn)*np.dot(BigClass.Variabs.alpha_vec, loss[0])

        # dual problem is different under schemes of change of R
        # if BigClass.Variabs.R_update == 'deltaR' and np.shape(self.input_dual_in_t)[0]>1:  # make sure not init value
        #     self.input_dual_nxt -= delta_p  # erase memory
        if BigClass.Variabs.R_update == 'propto':  # if resistances change with memory
            self.input_dual_nxt: NDArray[np.float_] = input_dual - delta
        elif BigClass.Variabs.R_update == 'deltaR':  # no memory
            self.input_dual_nxt = - delta
        self.input_dual_in_t.append(self.input_dual_nxt)  # append into list in time
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('loss=', loss)
            print('time=', self.t)
            print('input_dual_nxt=', self.input_dual_nxt)

    def update_output_dual(self, BigClass: "Big_Class"):
        """
        Calculates next output pressure values in dual problem given measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        output_dual_nxt: np.ndarray sized [Nout,] denoting output pressure of dual problem at time t
        """
        loss: NDArray[np.float_] = self.loss_in_t[-1]
        output_dual: NDArray[np.float_] = copy.copy(self.output_dual_in_t[-1])
        # element-wise multiplication for alpha in output update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            output_prev: NDArray[np.float_] = self.output_in_t[-2]
            delta: NDArray[np.float_] = BigClass.Variabs.alpha_vec * (self.output-output_prev) * (loss[0]-loss[1])
        else:
            delta = BigClass.Variabs.alpha_vec * self.output * loss[0]

        # dual problem is different under schemes of change of R
        if BigClass.Variabs.R_update == 'propto':  # if resistances change with memory
            self.output_dual_nxt = output_dual + delta
        elif BigClass.Variabs.R_update == 'deltaR':  # no memory
            self.output_dual_nxt = delta
        self.output_dual_in_t.append(self.output_dual_nxt)
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('output_dual_nxt', self.output_dual_nxt)

    def update_inter_dual(self, BigClass: "Big_Class") -> None:
        """
        Calculates next inter nodes pressure values in dual problem given measurement, for 1 or 2 sampled pressures
        only for when Variabs.access_interNodes==True

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        interNodes_dual_nxt: np.ndarray sized [Ninter,] denoting inter nodes pressure of dual problem at time t
        """
        loss: NDArray[np.float_] = self.loss_in_t[-1]  # copy loss
        inter_dual: NDArray[np.float_] = self.inter_dual_in_t[-1]
        inter: NDArray[np.float_] = self.inter_in_t[-1]
        # dot product for alpha in inter nodes pressure update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            inter_prev: NDArray[np.float_] = self.inter_in_t[-2]
            delta: NDArray[np.float_] = (inter-inter_prev)*np.dot(BigClass.Variabs.alpha_vec,
                                                                  loss[0]-loss[1])
        else:  # if one sample of p in for every loss calcaultion are to be taken
            delta = inter*np.dot(BigClass.Variabs.alpha_vec, loss[0])

        # dual problem is different under schemes of change of R
        if BigClass.Variabs.R_update == 'propto':  # if resistances change with memory
            # self.inter_dual_nxt = inter_dual - delta + 0.01*np.random.randn(BigClass.Variabs.Ninter)
            self.inter_dual_nxt = inter_dual - delta
        elif BigClass.Variabs.R_update == 'deltaR':  # no memory
            # self.inter_dual_nxt = - delta + 0.01*np.random.randn(BigClass.Variabs.Ninter)
            self.inter_dual_nxt = - delta
        self.inter_dual_in_t.append(self.inter_dual_nxt)  # append into list in time
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('inter_dual_nxt=', self.inter_dual_nxt)

    def update_Rs(self, BigClass: "Big_Class") -> None:
        """
        update resistances of NE edges

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        R_vec  - [NE] np.array of resistivities
        """
        R_vec: NDArray[np.float_] = self.R_in_t[-1]
        delta_p: NDArray[np.float_] = self.u * R_vec
        if BigClass.Variabs.R_update == 'deltaR':  # delta_R propto p_in-p_out
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * delta_p)
        elif BigClass.Variabs.R_update == 'propto':  # R propto p_in-p_out
            self.R_in_t.append(BigClass.Variabs.gamma * delta_p)
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            pass
            # print('R_nxt', self.R_in_t[-1])

    # def measure_targets_iris(self, BigClass):
    #     """
    #     """
    #     CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]  # type hint
    #     CstrTuple = functions.setup_constraints_given_pin(
    #                     (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.ground_nodes_arr),
    #                     self.input_drawn, BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
    #     self.output_means, self.u = solve.solve_flow(BigClass, CstrTuple, self.R_in_t[-1])
