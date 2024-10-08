from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import functions, solve, statistics

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
    def __init__(self, Variabs: "User_Variables") -> None:
        super().__init__()
        self.t: int = 0  # time, defined as number of R updates, i.e. times the learning rate alpha is used.
        self.p: NDArray[np.float_] = array([])  # pressure
        self.u: NDArray[np.float_] = array([])  # flow rate
        self.input_drawn_in_t: List[NDArray[np.float_]] = []  # pressure at inputs in time, sampled
        self.extraInput_in_t: List[NDArray[np.float_]] = []  # pressure at extra inputs in time
        self.inter_in_t: List[NDArray[np.float_]] = []
        # pressure at outputs and extra in time
        self.output_in_t: List[NDArray[np.float_]] = []
        self.extraOutput_in_t: List[NDArray[np.float_]] = []
        self.desired_in_t: List[NDArray[np.float_]] = []
        # inputs and extra inputs at dual problem in time
        self.input_dual_in_t: List[NDArray[np.float_]] = [2. * np.ones(Variabs.Nin)]
        self.extraInput_dual_in_t: List[NDArray[np.float_]] = [2. * np.ones(Variabs.extraNin)]
        self.inter_dual_in_t: List[NDArray[np.float_]] = [np.random.random(Variabs.Ninter)]
        # outputs and extra outputs during dual problem in time
        self.output_dual_in_t: List[NDArray[np.float_]] = [1. * np.ones(Variabs.Nout)]
        self.extraOutput_dual_in_t: List[NDArray[np.float_]] = [1. * np.ones(Variabs.extraNout)]
        self.loss_in_t: List[NDArray[np.float_]] = []
        self.loss_norm_in_t: List[NDArray[np.float_]] = []  # normalized loss
        self.Power_norm_in_t: List[NDArray[np.float_]] = []
        # Other sizes that make problems sometimes
        self.extraInput: NDArray[np.float_] = copy.copy(self.extraInput_dual_in_t[-1])

    def initiate_resistances(self, BigClass: "Big_Class", R_vec_i: Optional[NDArray[np.float_]] = None) -> None:
        """
        After using build_incidence, initiate resistances

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        R_vec_i  - initial resistances, array of size [NE,]
        """
        if R_vec_i is not None:
            if np.size(R_vec_i) != BigClass.Strctr.NE:
                print('R_vec_i has wrong size, initializing all ones')
                self.R_in_t: List[NDArray[np.float_]] = [np.ones((BigClass.Strctr.NE), dtype=float)]
            else:
                self.R_in_t = [R_vec_i]
        else:
            self.R_in_t = [np.ones((BigClass.Strctr.NE), dtype=float)]

    def initiate_accuracy_vec(self, BigClass: "Big_Class", measure_accuracy_every: int) -> None:
        """
        Initiate array for accuracy with length=iteration/measure_accuracy_every

        inputs:
        BigClass               - class instance including User_Variables, Network_Structure instances, etc.
        measure_accuracy_every - measure accuracy every # steps, user input
        """
        accuracy_size = int(np.floor(BigClass.Variabs.iterations/measure_accuracy_every))
        self.accuracy_in_t: NDArray[np.float_] = zeros(accuracy_size)
        self.t_for_accuracy: NDArray[np.int_] = zeros(accuracy_size, dtype=np.int_)

    def draw_p_in_and_desired(self, Variabs: "User_Variables", i: int, noise_to_extra: Optional[bool] = False,
                              problem: Optional[str] = "measure") -> None:
        """
        Every time step, draw random input pressures and calculate the desired output given input

        inputs:
        Variabs - User_Variables class
        i       - int, iteration #

        outputs
        input_drawn: np.ndarray sized [Nin,], input pressures
        desired: np.ndarray sized [Nout,], desired output defined by the task M*p_input
        """
        # draw  input from train or test sets
        if problem == 'measure_for_accuracy':
            self.input_drawn: NDArray[np.float_] = copy.copy(Variabs.X_test[i % np.shape(Variabs.X_test)[0]])
        else:
            self.input_drawn = copy.copy(Variabs.X_train[i % np.shape(Variabs.X_train)[0]])

        # draw noise if needed
        if noise_to_extra:
            self.extraInput += Variabs.noise_in[i % np.shape(Variabs.noise_in)[0]]
            self.extraOutput: NDArray[np.float_] = copy.copy(self.extraOutput_in_t[-1])
            self.extraOutput += Variabs.noise_out[i % np.shape(Variabs.noise_out)[0]]
            self.inter: NDArray[np.float_] = copy.copy(self.inter) + \
                Variabs.noise_inter[i % np.shape(Variabs.noise_inter)[0]]

        # calculate desired output from train or test sets
        if Variabs.task_type == 'Iris_classification':
            if problem == 'measure_for_accuracy':
                self.desired: NDArray[np.float_] = np.matmul(Variabs.y_test[i % np.shape(Variabs.X_test)[0]],
                                                             self.targets_mat)
            else:
                self.desired = np.matmul(Variabs.y_train[i % np.shape(Variabs.X_train)[0]],
                                         self.targets_mat)
        else:
            self.desired = Variabs.y_train[i % np.shape(Variabs.X_train)[0]]

        # append to arrays in time
        if problem == 'measure_for_accuracy':  # don't add to time vector if this is accuracy calculation
            pass
        else:
            self.input_drawn_in_t.append(self.input_drawn)
            self.extraInput_in_t.append(self.extraInput)
            self.desired_in_t.append(self.desired)

        # optionally print to user
        if Variabs.supress_prints:
            pass
        else:  # print
            print('input_drawn', self.input_drawn)
            # print('extraInput', self.extraInput)
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

    def assign_targets_Iris(self, BigClass: "Big_Class") -> None:
        """
        """
        targets_mat: NDArray[np.float_] = zeros([3, 3], dtype=np.float_)
        for j in range(3):  # go over all 3 Iris classes
            self.draw_p_means_Iris(BigClass.Variabs, j)  # take the mean of all data inputs of a specific Iris class
            # measure output while input is mean, don't change resistances
            self.solve_flow_given_problem(BigClass, "measure_for_mean")
            targets_mat[j] = self.output  # The new target is the outputs of the mean input
        self.targets_mat: NDArray[np.float_] = targets_mat  # save into targets_mat array
        if BigClass.Variabs.supress_prints:  # don't print outputs
            pass
        else:  # print
            print('targets_mat', self.targets_mat)

    def solve_flow_given_problem(self, BigClass: "Big_Class", problem: str,
                                 noise_to_extra: Optional[bool] = False, access_inters: Optional[bool] = False) -> None:
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
        if problem == 'measure' or problem == 'measure_for_mean' or problem == 'measure_for_accuracy':
            if noise_to_extra:
                CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]  # type hint
                CstrTuple = functions.setup_constraints_given_pin(
                            (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                             BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.inter_nodes_arr),
                            (self.input_drawn, self.extraInput, self.inter), BigClass.Strctr.NN,
                            BigClass.Strctr.EI, BigClass.Strctr.EJ)
            else:
                CstrTuple = functions.setup_constraints_given_pin(
                            (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                             BigClass.Strctr.ground_nodes_arr),
                            (self.input_drawn, self.extraInput), BigClass.Strctr.NN, BigClass.Strctr.EI,
                            BigClass.Strctr.EJ)
        elif problem == 'dual':
            if BigClass.Variabs.access_interNodes or access_inters:  # if dual problem accesses interNodes separately
                CstrTuple = functions.setup_constraints_given_pin(
                            (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                             BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.output_nodes_arr,
                             BigClass.Strctr.extraOutput_nodes_arr, BigClass.Strctr.inter_nodes_arr),
                            (self.input_dual_in_t[-1], self.extraInput_dual_in_t[-1], self.output_dual_in_t[-1],
                             self.extraOutput_dual_in_t[-1], self.inter_dual_in_t[-1]),
                            BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
            else:  # if dual problem does not access interNodes separately
                CstrTuple = functions.setup_constraints_given_pin(
                            (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                             BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.output_nodes_arr,
                             BigClass.Strctr.extraOutput_nodes_arr),
                            (self.input_dual_in_t[-1], self.extraInput_dual_in_t[-1], self.output_dual_in_t[-1],
                             self.extraOutput_dual_in_t[-1]),
                            BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
        # print('CStr Tuple', CstrTuple)
        self.p, self.u = solve.solve_flow(BigClass, CstrTuple, self.R_in_t[-1])

        # Update the State class variables
        if problem in {'measure', 'measure_for_mean', 'measure_for_accuracy'}:
            self.inter = copy.copy(self.p[BigClass.Strctr.inter_nodes_arr].ravel())
            self.output: NDArray[np.float_] = copy.copy(self.p[BigClass.Strctr.output_nodes_arr].ravel())
            self.extraOutput = copy.copy(self.p[BigClass.Strctr.extraOutput_nodes_arr].ravel())
            if BigClass.Variabs.supress_prints:
                pass
            else:  # print
                # print('inter measured=', self.inter)
                print('output measured=', self.output)
                # print('extraOutput measured=', self.extraOutput)

            if problem == 'measure':  # Only save in time if measuring during training
                self.output_in_t.append(self.output)
                self.extraOutput_in_t.append(self.extraOutput)
                self.inter_in_t.append(self.inter)

    def update_input_dual(self, BigClass: "Big_Class") -> None:
        """
        Calculates next input pressure values in dual problem given the measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        input_dual_nxt: np.ndarray sized [Nin,] denoting input pressure of dual problem at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
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
        if R_update == 'R_propto_dp' or R_update == 'R_propto_Q':  # R changes with memory
            self.input_dual_nxt: NDArray[np.float_] = input_dual - delta
        # else if no memory
        elif R_update == 'deltaR_propto_dp' or R_update == 'deltaR_propto_Q' or R_update == 'deltaR_propto_Power':
            self.input_dual_nxt = - delta
        self.input_dual_in_t.append(self.input_dual_nxt)  # append into list in time
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('input_dual_nxt=', self.input_dual_nxt)

    def update_extraInput_dual(self, BigClass: "Big_Class"):
        """
        Calculates next pressure values for extra input nodes in dual problem given measurement,
        either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        extraInput_dual_nxt: np.ndarray sized [Nout,] denoting output pressure of dual problem at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]  # copy loss
        extraInput_dual: NDArray[np.float_] = self.extraInput_dual_in_t[-1]
        # print('extraInput_dual', extraInput_dual)
        extraInput: NDArray[np.float_] = self.extraInput_in_t[-1]
        # print('extraInput', extraInput)
        # dot product for alpha in pressure update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            extraInput_prev: NDArray[np.float_] = self.extraInput_in_t[-2]
            # print('extraInput_prev', extraInput_prev)
            delta: NDArray[np.float_] = (extraInput-extraInput_prev)*np.dot(BigClass.Variabs.alpha_vec,
                                                                            loss[0]-loss[1])
            # print('delta extra input', delta)
        else:  # if one sample of p in for every loss calcaultion are to be taken
            delta = (extraInput)*np.dot(BigClass.Variabs.alpha_vec, loss[0])

        # dual problem is different under schemes of change of R
        if R_update == 'R_propto_dp' or R_update == 'R_propto_Q':  # R changes with memory
            self.extraInput_dual_nxt: NDArray[np.float_] = extraInput_dual - delta
        # else if no memory
        elif R_update == 'deltaR_propto_dp' or R_update == 'deltaR_propto_Q' or R_update == 'deltaR_propto_Power':
            self.extraInput_dual_nxt = - delta
        self.extraInput_dual_in_t.append(self.extraInput_dual_nxt)  # append into list in time
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('extraInput_dual_nxt=', self.extraInput_dual_nxt)

    def update_inter_dual(self, BigClass: "Big_Class") -> None:
        """
        Calculates next inter nodes pressure values in dual problem given measurement, for 1 or 2 sampled pressures
        only for when Variabs.access_interNodes==True

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        interNodes_dual_nxt: np.ndarray sized [Ninter,] denoting inter nodes pressure of dual problem at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
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
        if R_update == 'R_propto_dp' or R_update == 'R_propto_Q':  # R changes with memory
            # self.inter_dual_nxt = inter_dual - delta + 0.01*np.random.randn(BigClass.Variabs.Ninter)
            self.inter_dual_nxt = inter_dual - delta
        # else if no memory
        elif R_update == 'deltaR_propto_dp' or R_update == 'deltaR_propto_Q' or R_update == 'deltaR_propto_Power':
            # self.inter_dual_nxt = - delta + 0.01*np.random.randn(BigClass.Variabs.Ninter)
            self.inter_dual_nxt = - delta
        self.inter_dual_in_t.append(self.inter_dual_nxt)  # append into list in time
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('inter_dual_nxt=', self.inter_dual_nxt)

    def update_output_dual(self, BigClass: "Big_Class"):
        """
        Calculates next output pressure values in dual problem given measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        output_dual_nxt: np.ndarray sized [Nout,] denoting output pressure of dual problem at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]
        output_dual: NDArray[np.float_] = copy.copy(self.output_dual_in_t[-1])
        # element-wise multiplication for alpha in output update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            output_prev: NDArray[np.float_] = self.output_in_t[-2]
            delta: NDArray[np.float_] = BigClass.Variabs.alpha_vec * (self.output-output_prev) * (loss[0]-loss[1])
        else:
            delta = BigClass.Variabs.alpha_vec * self.output * loss[0]

        # dual problem is different under schemes of change of R
        if R_update == 'R_propto_dp' or R_update == 'R_propto_Q':  # R changes with memory
            self.output_dual_nxt = output_dual + delta
        # else if no memory
        elif R_update == 'deltaR_propto_dp' or R_update == 'deltaR_propto_Q' or R_update == 'deltaR_propto_Power':
            self.output_dual_nxt = delta
        self.output_dual_in_t.append(self.output_dual_nxt)
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('output_dual_nxt', self.output_dual_nxt)

    def update_extraOutput_dual(self, BigClass: "Big_Class"):
        """
        Calculates next output pressure values in dual problem given measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        extraOutput_dual_nxt: np.ndarray sized [Nout,] denoting output pressure of dual problem at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]
        extraOutput_dual: NDArray[np.float_] = copy.copy(self.extraOutput_dual_in_t[-1])
        # element-wise multiplication for alpha in output update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            extraOutput_prev: NDArray[np.float_] = self.extraOutput_in_t[-2]
            delta: NDArray[np.float_] = (self.extraOutput-extraOutput_prev) * np.dot(BigClass.Variabs.alpha_vec,
                                                                                     loss[0]-loss[1])
        else:
            delta = self.extraOutput * np.dot(BigClass.Variabs.alpha_vec, loss[0])
        # dual problem is different under schemes of change of R
        if R_update == 'R_propto_dp' or R_update == 'R_propto_Q':  # R changes with memory
            self.extraOutput_dual_nxt = extraOutput_dual + delta
        # else if no memory
        elif R_update == 'deltaR_propto_dp' or R_update == 'deltaR_propto_Q' or R_update == 'deltaR_propto_Power':
            self.extraOutput_dual_nxt = delta
        self.extraOutput_dual_in_t.append(self.extraOutput_dual_nxt)
        # if user ask to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            print('extraOutput_dual_nxt', self.extraOutput_dual_nxt)

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
        if BigClass.Variabs.R_update == 'deltaR_propto_dp':  # delta_R propto p_in-p_out
            # self.R_in_t.append(np.abs(R_vec + BigClass.Variabs.gamma * delta_p))
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * delta_p)
        elif BigClass.Variabs.R_update == 'R_propto_dp':  # R propto p_in-p_out
            # self.R_in_t.append(BigClass.Variabs.gamma * np.abs(delta_p))
            self.R_in_t.append(BigClass.Variabs.gamma * delta_p)
        elif BigClass.Variabs.R_update == 'deltaR_propto_Q':  # delta_R propto flow Q
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * self.u)
        elif BigClass.Variabs.R_update == 'R_propto_Q':  # R propto flow Q
            self.R_in_t.append(BigClass.Variabs.gamma * self.u)
        elif BigClass.Variabs.R_update == 'deltaR_propto_Power':  # delta_R propto Power dissipation dp*Q
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * self.u * delta_p * np.sign(delta_p))
        # if user asks to not print
        if BigClass.Variabs.supress_prints:
            pass
        else:  # print
            pass
            # print('R_nxt', self.R_in_t[-1])
        # self.R_in_t[-1][BigClass.Strctr.EJ == BigClass.Strctr.ground_nodes_arr] = 1.

    def calc_loss(self, BigClass: "Big_Class") -> None:
        """
        Calculates the loss given system state and desired outputs, perhaps including 1 time step ago

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        loss: np.ndarray sized [Nout,]
        """
        if BigClass.Variabs.loss_fn == functions.loss_fn_2samples:
            if BigClass.Variabs.include_Power:
                self.loss: NDArray[np.float_] = BigClass.Variabs.loss_fn(self.output, self.output_in_t[-2],
                                                                         self.desired, self.desired_in_t[-2],
                                                                         self.Power_norm, self.Power_norm_in_t[-2],
                                                                         BigClass.Variabs.lam)
            else:
                self.loss: NDArray[np.float_] = BigClass.Variabs.loss_fn(self.output, self.output_in_t[-2],
                                                                         self.desired, self.desired_in_t[-2])
        elif BigClass.Variabs.loss_fn == functions.loss_fn_1sample:
            if BigClass.Variabs.include_Power:
                print('Power_norm', self.Power_norm)
                print('lam', BigClass.Variabs.lam)
                self.loss = BigClass.Variabs.loss_fn(self.output, self.desired, self.Power_norm, BigClass.Variabs.lam)
            else:
                self.loss = BigClass.Variabs.loss_fn(self.output, self.desired)
        self.loss_norm = self.loss/np.mean(np.abs(self.desired))
        
        self.loss_in_t.append(self.loss)
        self.loss_norm_in_t.append(self.loss_norm)

    def calc_Power_norm(self, BigClass: "Big_Class"):
        self.Power_norm = statistics.power_dissip_norm(self.u, self.R_in_t[-1], self.input_drawn)
        self.Power_norm_in_t.append(self.Power_norm)
        if BigClass.Variabs.supress_prints:
            pass
        else:
            print('Power dissipation normalized', self.Power_norm)

    def calculate_accuracy_fullDataset(self, BigClass: "Big_Class") -> None:
        self.accuracy_vec: NDArray[np.int_] = zeros(np.shape(BigClass.Variabs.dataset)[0], dtype=np.int_)
        for i, datapoint in enumerate(BigClass.Variabs.dataset):
            self.draw_p_in_and_desired(BigClass.Variabs, i, problem='measure_for_accuracy')
            self.solve_flow_given_problem(BigClass, "measure_for_accuracy")  # measure and don't change resistances
            # print('net prediction', np.sum((self.targets_mat - self.output)**2, axis=1))
            self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
                                                                         BigClass.Variabs.targets[i])
        self.accuracy = np.mean(self.accuracy_vec)

    def calculate_accuracy_testset(self, BigClass: "Big_Class") -> None:
        self.accuracy_vec = zeros(np.shape(BigClass.Variabs.X_test)[0], dtype=np.int_)
        for i, datapoint in enumerate(BigClass.Variabs.X_test):
            self.draw_p_in_and_desired(BigClass.Variabs, i, problem='measure_for_accuracy')
            self.solve_flow_given_problem(BigClass, "measure_for_accuracy")  # measure and don't change resistances
            # self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
            #                                                              BigClass.Variabs.targets[i])
            self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
                                                                         BigClass.Variabs.y_test[i])
        self.accuracy = np.mean(self.accuracy_vec)
