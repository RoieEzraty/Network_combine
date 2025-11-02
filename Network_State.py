from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import functions, solve, statistics, matrix_functions

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Big_Class import Big_Class
    from Network_Structure import Network_Structure


# ===================================================
# Class - network state variables
# ===================================================


class Network_State:
    """
    Class with variables that hold information of state of network.
    what ends with _in_t holds all time instances of the variable, each list index is different t
    what ends w/out _in_t is at current time instance self.t
    """
    def __init__(self, Variabs: "User_Variables", input_update_initial: NDArray[np.float_] = array([]),
                 output_update_initial: NDArray[np.float_] = array([])) -> None:
        super().__init__()
        self.t: int = 0  # time, defined as number of R updates, i.e. times the learning rate alpha is used.
        self.p: NDArray[np.float_] = array([])  # pressure
        self.u: NDArray[np.float_] = array([])  # flow rate
        # "measurement" modality
        self.input_drawn_in_t: List[NDArray[np.float_]] = []  # pressure at inputs in time
        self.extraInput_in_t: List[NDArray[np.float_]] = []  # pressure at additional inputs in time
        self.inter_in_t: List[NDArray[np.float_]] = []  # pressure at intermediate nodes (not input/output) in time
        self.output_in_t: List[NDArray[np.float_]] = []  # pressure at outputs in time
        self.extraOutput_in_t: List[NDArray[np.float_]] = []  # pressure at additional outputs, loss not calculated
        self.desired_in_t: List[NDArray[np.float_]] = []  # desired output pressure for each sample, input dependent
        # "update" modality
        if input_update_initial.size:
            self.input_update_in_t: List[NDArray[np.float_]] = [input_update_initial]
        else:
            self.input_update_in_t = [1. * np.ones(Variabs.Nin)]
        self.extraInput_update_in_t: List[NDArray[np.float_]] = [1. * np.ones(Variabs.extraNin)]
        self.inter_update_in_t: List[NDArray[np.float_]] = [np.random.random(Variabs.Ninter)]
        if output_update_initial.size:
            self.output_update_in_t: List[NDArray[np.float_]] = [output_update_initial]
        else:
            self.output_update_in_t = [0.5 * np.ones(Variabs.Nout)]
        self.extraOutput_update_in_t: List[NDArray[np.float_]] = [0.5 * np.ones(Variabs.extraNout)]
        if Variabs.hysteresis:
            self.hysteresis = 0
        # learning rate
        self.alpha_in_t = Variabs.alpha_vec[0]*np.ones([Variabs.iterations])
        self.alpha: np.float_ = Variabs.alpha_vec[0]
        # Loss and Power
        self.loss_in_t: List[NDArray[np.float_]] = []
        self.loss_scalar_in_t: List[NDArray[np.float_]] = []
        self.Power_norm_in_t: List[NDArray[np.float_]] = []  # Power dissipation in whole network, normalized by inputs
        self.update_vec_in_t: List[NDArray[np.float_]] = []  # input and output values in update modality, w/out hysteresis
        # Other sizes that make problems sometimes
        self.extraInput: NDArray[np.float_] = copy.copy(self.extraInput_update_in_t[-1])

    def initiate_resistances(self, BigClass: "Big_Class", R_vec_i: Optional[NDArray[np.float_]] = None,
                             add_noise: Optional[float] = 0.0) -> None:
        """
        After using build_incidence, initiate resistances

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        R_vec_i  - optional initial resistances, array of size [NE,]
        """
        if R_vec_i is not None:  # user speficied initial resistances
            if np.size(R_vec_i) != BigClass.Strctr.NE:
                print('R_vec_i has wrong size, initializing all ones')
                self.R_in_t: List[NDArray[np.float_]] = [np.ones((BigClass.Strctr.NE), dtype=float)]
            else:
                self.R_in_t = [R_vec_i]
        else:  # uniform resistances - not user specified
            self.R_in_t = [np.ones((BigClass.Strctr.NE), dtype=float)]

        if add_noise:
            self.R_in_t[0] += np.random.normal(loc=0.0, scale=add_noise, size=BigClass.Strctr.NE)
        # resistances for bead net as if w/out beads
        self.R_backg: NDArray[np.float_] = BigClass.Variabs.R_min * np.ones(BigClass.Strctr.NE)

    def initiate_accuracy_vec(self, BigClass: "Big_Class", measure_accuracy_every: int) -> None:
        """
        For classification task, initiate array for accuracy with length=iteration/measure_accuracy_every

        inputs:
        BigClass               - class instance including User_Variables, Network_Structure instances, etc.
        measure_accuracy_every - measure accuracy every # steps, user input
        """
        accuracy_size = int(np.floor(BigClass.Variabs.iterations/measure_accuracy_every))
        self.accuracy_in_t: NDArray[np.float_] = zeros(accuracy_size)
        self.t_for_accuracy: NDArray[np.int_] = zeros(accuracy_size, dtype=np.int_)

    def draw_p_in_and_desired(self, Variabs: "User_Variables", i: int, noise_to_extra: Optional[bool] = False,
                              modality: Optional[str] = "measure") -> None:
        """
        Every time step, draw random input pressures and calculate the desired output given input

        inputs:
        Variabs        - User_Variables class
        i              - int, iteration #
        noise_to_extra - bool, whether to add noise to p on extra nodes
        modality       - str, "measure" for measurement modality where outputs are measured
                              "update" for update modality where outputs are constained
                              and after which resistances change

        outputs
        input_drawn: np.ndarray sized [Nin,], input pressures
        desired: np.ndarray sized [Nout,], desired output defined by the task M*p_input
        """
        # draw  input from train or test sets
        if modality == 'measure_for_accuracy':
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
            if modality == 'measure_for_accuracy':
                self.desired: NDArray[np.float_] = \
                    np.matmul(Variabs.y_test[i % np.shape(Variabs.X_test)[0]], self.targets_mat)
            else:
                self.desired = \
                    np.matmul(Variabs.y_train[i % np.shape(Variabs.X_train)[0]], self.targets_mat)
        else:
            self.desired = Variabs.y_train[i % np.shape(Variabs.X_train)[0]]

        # append to arrays in time
        if modality == 'measure_for_accuracy':  # don't add to time vector if this is accuracy calculation
            pass
        else:
            self.input_drawn_in_t.append(self.input_drawn)
            self.extraInput_in_t.append(self.extraInput)
            self.desired_in_t.append(self.desired)

        # optionally print to user
        if not Variabs.supress_prints:
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
        Compute and assign class-specific output targets for the Iris classification task.

        For each of the 3 Iris classes:
        - Computes the mean of the input data belonging to that class.
        - Simulates the network flow for that class-specific input mean.
        - Stores the resulting output as the target for that class.

        Parameters
        ----------
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        """
        targets_mat: NDArray[np.float_] = zeros([3, 3], dtype=np.float_)
        for j in range(3):  # go over all 3 Iris classes
            self.draw_p_means_Iris(BigClass.Variabs, j)  # compute class-mean input for class j
            self.solve_flow_given_modality(BigClass, "measure_for_mean")  # simulate without changing resistances
            targets_mat[j] = self.output  # The new target is the outputs of the mean input
        self.targets_mat: NDArray[np.float_] = targets_mat  # save into targets_mat array

        # optionally print to user
        if not BigClass.Variabs.supress_prints:
            print('targets_mat', self.targets_mat)

    def solve_flow_given_modality(self, BigClass: "Big_Class", modality: str,
                                  noise_to_extra: Optional[bool] = False,
                                  access_inters: Optional[bool] = False) -> None:
        """
        Calculates the constraint matrix Cstr, then solves the flow,
        using functions from functions.py and solve.py,
        given the modality variable.

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        modality - string stating the modality type: "measure" for no constraint on outputs
                                                     "measure_for_mean" for outputs of mean of Iris class
                                                     "measure_for_accuracy" for outputs of mean of Iris class
                                                     "update" for constrained outputs as well
        noise_to_extra - optional bool, whether to add noise to p on extra nodes
        access_inters  - optional bool, whether to change pressure in inter nodes

        outputs:
        p - pressure at every node under the specific BC, after convergence while allowing conductivities to change
        u - flow at every edge under the specific BC, after convergence while allowing conductivities to change
        """
        # Calculate pressure p and flow u
        # Select nodes and pressure data based on modality
        nodes_tuple: Union[Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]],
                           Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]],
                           Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                 NDArray[np.int_]],
                           Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
                                 NDArray[np.int_], NDArray[np.int_]]]
        nodeData_tuple: Union[Tuple[NDArray[np.float_], NDArray[np.float_]],
                              Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
                              Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
                              Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_],
                                    NDArray[np.float_]]]
        if modality in {'measure', 'measure_for_mean', 'measure_for_accuracy'}:
            if noise_to_extra:
                nodes_tuple = (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                               BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.inter_nodes_arr)
                nodeData_tuple = (self.input_drawn, self.extraInput, self.inter)
            else:
                nodes_tuple = (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                               BigClass.Strctr.ground_nodes_arr)
                nodeData_tuple = (self.input_drawn, self.extraInput)
        elif modality == 'update':
            # Access inter nodes if needed
            inters = BigClass.Variabs.access_interNodes or access_inters

            nodes_tuple = (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                           BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.output_nodes_arr,
                           BigClass.Strctr.extraOutput_nodes_arr)
            nodeData_tuple = (self.input_update_in_t[-1], self.extraInput_update_in_t[-1],
                              self.output_update_in_t[-1], self.extraOutput_update_in_t[-1])

            if inters:  # add inter nodes if needed
                nodes_tuple += tuple([BigClass.Strctr.inter_nodes_arr])
                nodeData_tuple += tuple([self.inter_update_in_t[-1]])
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Constraint matrix given constrained nodes and values
        self.CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]
        self.CstrTuple = functions.setup_constraints_given_pin(nodes_tuple, nodeData_tuple, BigClass.Strctr.NN,
                                                               BigClass.Strctr.EI, BigClass.Strctr.EJ)

        # R to K
        self.K_vec: NDArray[np.float_]  # type hint conductivities
        self.K_vec = matrix_functions.K_from_R(self.R_in_t[-1])  # calculate conductivities

        self.p, self.u = solve.solve_flow(BigClass.Strctr, self.CstrTuple, self.K_vec)

        # add to State class variables
        if modality in {'measure', 'measure_for_mean', 'measure_for_accuracy'}:
            self.inter = copy.copy(self.p[BigClass.Strctr.inter_nodes_arr].ravel())
            self.output: NDArray[np.float_] = copy.copy(self.p[BigClass.Strctr.output_nodes_arr].ravel())
            self.extraOutput = copy.copy(self.p[BigClass.Strctr.extraOutput_nodes_arr].ravel())

            # print
            if not BigClass.Variabs.supress_prints:
                # print('inter measured=', self.inter)
                print('output measured=', self.output)
                # print('extraOutput measured=', self.extraOutput)

            if modality == 'measure':  # Only save in time if measuring during training
                self.output_in_t.append(self.output)
                self.extraOutput_in_t.append(self.extraOutput)
                self.inter_in_t.append(self.inter)

    def update_alpha(self, BigClass: "Big_Class", T_annealing: float) -> None:
        """
        update learning rate alpha for annealing scheme

        inputs:
        BigClass    - Class instance containing User_Variables, Network_Structure, etc.
        T_annealing - float, exponent time of annealing. larger T = slower annealing
        """
        self.alpha = BigClass.Variabs.alpha_vec[0] * \
            np.exp(-BigClass.State.t / (T_annealing * BigClass.Variabs.iterations))
        # np.cos(np.pi * BigClass.State.t / BigClass.Variabs.T_annealing)**2
        self.alpha_in_t[self.t] = self.alpha

    def update_input(self, BigClass: "Big_Class") -> None:
        """
        Calculates next input pressure values in update modality given measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass - Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        input_update_nxt: np.ndarray [Nin,] input pressure of update modality at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]  # copy loss
        input_update: NDArray[np.float_] = self.input_update_in_t[-1]
        input_drawn: NDArray[np.float_] = self.input_drawn_in_t[-1]

        if BigClass.Variabs.training_scheme in ['GD_like', 'Adaline']:
            delta: NDArray[np.float_] = - self.update_vec[BigClass.Strctr.input_nodes_arr]
        else:
            if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
                input_drawn_prev: NDArray[np.float_] = self.input_drawn_in_t[-2]
            else:  # use zero input, output and loss for 2nd sample
                input_drawn_prev = np.zeros([BigClass.Variabs.Nin])
                loss = np.array([copy.copy(loss[0]), np.zeros([BigClass.Variabs.Nout])])  # good loss dims for next "if"
            if BigClass.Variabs.normalize_loss:
                delta = (input_drawn-input_drawn_prev) * self.alpha * \
                    (np.mean(loss[0]-loss[1])/np.linalg.norm(loss[0]-loss[1]))
            else:
                delta = (input_drawn-input_drawn_prev) * self.alpha * np.mean(loss[0]-loss[1])

        # update modality is different under schemes of change of R

        # w/ memory
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp']:
            self.input_update_nxt: NDArray[np.float_] = input_update - delta
        elif R_update == 'beads':
            self.input_update_nxt = input_update + self.alpha * np.mean(np.abs(loss[0]))
        # else if no memory
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.input_update_nxt = - delta
        elif R_update == 'grad_desc':
            self.input_update_nxt = input_update

        # reset "update" modality values if diverging
        if functions.reset_update(self.input_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.input_update_nxt = self.input_update_in_t[0]

        self.input_update_in_t.append(self.input_update_nxt)  # append into list in time

        # print
        if not BigClass.Variabs.supress_prints:
            print('input_update_nxt=', self.input_update_nxt)

    def update_extraInput(self, BigClass: "Big_Class"):
        """
        Calculates next pressure values for extra input nodes in update modality given measurement,
        either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        extraInput_update_nxt: np.ndarray sized [Nout,] denoting output pressure of update modality at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]  # copy loss
        extraInput_update: NDArray[np.float_] = self.extraInput_update_in_t[-1]
        extraInput: NDArray[np.float_] = self.extraInput_in_t[-1]

        # dot product for alpha in pressure update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            extraInput_prev: NDArray[np.float_] = self.extraInput_in_t[-2]
            delta: NDArray[np.float_] = (extraInput-extraInput_prev) * self.alpha * np.mean(loss[0]-loss[1])
        else:  # if one sample of p in for every loss calcaultion are to be taken
            delta = extraInput * self.alpha * np.mean(loss[0])

        # update modality is different under schemes of change of R

        # w/ memory
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp', 'beads']:
            self.extraInput_update_nxt: NDArray[np.float_] = extraInput_update - delta
        # else if no memory
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.extraInput_update_nxt = - delta
        elif R_update == 'grad_desc':
            self.extraInput_update_nxt = extraInput_update

        # reset "update" modality values if diverging
        if functions.reset_update(self.extraInput_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.extraInput_update_nxt = self.extraInput_update_in_t[0]
        self.extraInput_update_in_t.append(self.extraInput_update_nxt)  # append into list in time

        # print
        if not BigClass.Variabs.supress_prints:
            print('extraInput_update_nxt=', self.extraInput_update_nxt)

    def update_inter(self, BigClass: "Big_Class") -> None:
        """
        Calculates next inter nodes pressure values in update modality given measurement, for 1 or 2 sampled pressures
        only for when Variabs.access_interNodes==True

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        interNodes_update_nxt: np.ndarray sized [Ninter,] denoting inter nodes pressure of update modality at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]  # copy loss
        inter_update: NDArray[np.float_] = self.inter_update_in_t[-1]
        inter: NDArray[np.float_] = self.inter_in_t[-1]

        # dot product for alpha in inter nodes pressure update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            inter_prev: NDArray[np.float_] = self.inter_in_t[-2]
            delta: NDArray[np.float_] = (inter-inter_prev) * self.alpha * np.mean(loss[0]-loss[1])
        else:  # if one sample of p in for every loss calcaultion are to be taken
            delta = inter * self.alpha * np.mean(loss[0])

        # update modality is different under schemes of change of R

        # w/ memory
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp', 'beads']:
            # self.inter_update_nxt = inter_update - delta + 0.01*np.random.randn(BigClass.Variabs.Ninter)
            self.inter_update_nxt = inter_update - delta
        # else if no memory
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            # self.inter_update_nxt = - delta + 0.01*np.random.randn(BigClass.Variabs.Ninter)
            self.inter_update_nxt = - delta
        elif R_update == 'grad_desc':
            self.inter_update_nxt = inter_update

        # reset "update" modality values if diverging
        if functions.reset_update(self.inter_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.inter_update_nxt = self.inter_update_in_t[0]

        self.inter_update_in_t.append(self.inter_update_nxt)  # append into list in time

        # print
        if not BigClass.Variabs.supress_prints:
            print('inter_update_nxt=', self.inter_update_nxt)

    def update_output(self, BigClass: "Big_Class"):
        """
        Calculates next output pressure values in update modality given measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        output_update_nxt: np.ndarray sized [Nout,] denoting output pressure of update modality at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]
        output_update: NDArray[np.float_] = copy.copy(self.output_update_in_t[-1])

        # element-wise multiplication for alpha in output update
        if BigClass.Variabs.training_scheme in ['GD_like', 'Adaline']:
            delta: NDArray[np.float_] = self.update_vec[BigClass.Strctr.output_nodes_arr]
        else:
            if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
                output_prev: NDArray[np.float_] = self.output_in_t[-2]
                if BigClass.Variabs.normalize_loss:
                    loss_multip = (loss[0]-loss[1])/np.linalg.norm(loss[0]-loss[1])
                else:
                    loss_multip = loss[0]-loss[1]
                delta = self.alpha * (self.output-output_prev) * loss_multip  # normalize loss
            else:
                if BigClass.Variabs.normalize_loss:
                    loss_multip = loss[0]/np.linalg.norm(loss[0])
                else:
                    loss_multip = loss[0]
                delta = self.alpha * self.output * loss_multip  # normalize loss
                
        # update modality is different under schemes of change of R

        # w/ memory
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp']:
            self.output_update_nxt = output_update + delta
        elif R_update == 'beads':
            self.output_update_nxt = output_update + self.alpha * np.mean(loss[0])
        # else if no memory
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.output_update_nxt = delta
        elif R_update == 'grad_desc':
            self.output_update_nxt = output_update

        # reset "update" modality values if diverging
        if functions.reset_update(self.output_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.output_update_nxt = self.output_update_in_t[0]

        self.output_update_in_t.append(self.output_update_nxt)

        # print
        if not BigClass.Variabs.supress_prints:
            print('output_update_nxt', self.output_update_nxt)

    def update_extraOutput(self, BigClass: "Big_Class"):
        """
        Calculates next output pressure values in update modality given measurement, either for 1 or 2 sampled pressures

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        extraOutput_update_nxt: np.ndarray sized [Nout,] denoting output pressure of update modality at time t
        """
        R_update: str = BigClass.Variabs.R_update  # dummy variable
        loss: NDArray[np.float_] = self.loss_in_t[-1]
        extraOutput_update: NDArray[np.float_] = copy.copy(self.extraOutput_update_in_t[-1])
        # element-wise multiplication for alpha in output update
        if BigClass.Variabs.use_p_tag:  # if two samples of p in for every loss calcaultion are to be taken
            extraOutput_prev: NDArray[np.float_] = self.extraOutput_in_t[-2]
            delta: NDArray[np.float_] = (self.extraOutput-extraOutput_prev) * self.alpha * np.mean(loss[0]-loss[1])
        else:
            delta = self.extraOutput * self.alpha * np.mean(loss[0])
        # update modality is different under schemes of change of R

        # w/ memory
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp', 'beads']:
            self.extraOutput_update_nxt = extraOutput_update + delta
        # else if no memory
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.extraOutput_update_nxt = delta
        elif R_update == 'grad_desc':
            self.extraOutput_update_nxt = extraOutput_update

        # reset "update" modality values if diverging
        if functions.reset_update(self.extraOutput_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.extraOutput_update_nxt = self.extraOutput_update_in_t[0]

        self.extraOutput_update_in_t.append(self.extraOutput_update_nxt)

        # print
        if not BigClass.Variabs.supress_prints:
            print('extraOutput_update_nxt', self.extraOutput_update_nxt)

    def update_Rs(self, BigClass: "Big_Class", delta_K=[]) -> None:
        """
        update resistances of NE edges

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        R_vec  - [NE] array of resistivities
        """
        R_vec: NDArray[np.float_] = self.R_in_t[-1]
        delta_p: NDArray[np.float_] = self.u * R_vec
        # delta_p: NDArray[np.float_] = np.matmul(BigClass.Strctr.DM, BigClass.State.p[:BigClass.Strctr.NN])  # same as u * R_vec

        # if hysteretic material - update only if new history. if not hysteretic, update for sure
        if BigClass.Variabs.hysteresis:
            BigClass.State.hysteresis += delta_p
            update_cond: bool = ((BigClass.State.hysteresis > BigClass.Variabs.hyst_thresh) | 
                                 (BigClass.State.hysteresis < -BigClass.Variabs.hyst_thresh))
        else:
            update_cond = np.ones(BigClass.Strctr.NE)

        if BigClass.Variabs.R_update in {'deltaR_propto_dp', 'deltaR_propto_dp_decay'}:  # delta_R propto p_in-p_out
            delta_R = BigClass.Variabs.gamma*delta_p * update_cond
            if BigClass.Variabs.normalize_step:
                delta_R_norm = self.alpha * delta_R / np.linalg.norm(delta_R)
                R_nxt: NDArray[np.float_] = self.R_in_t[-1] + delta_R_norm
            else:
                if BigClass.Variabs.R_update == 'deltaR_propto_dp_decay':  # update and add decay of resistance to 1
                    R_nxt = self.R_in_t[-1] + delta_R - BigClass.Variabs.decay*(self.R_in_t[-1] - 1)
                else:  # update regular
                    R_nxt = self.R_in_t[-1] + delta_R
            self.R_in_t.append(np.clip(R_nxt, 1e-12, None))
        elif BigClass.Variabs.R_update == 'R_propto_dp':  # R propto p_in-p_out
            self.R_in_t.append(BigClass.Variabs.gamma * np.abs(delta_p))
            # self.R_in_t.append(BigClass.Variabs.gamma * delta_p)
        elif BigClass.Variabs.R_update == "R_propto_sqrt_dp":
            self.R_in_t.append(BigClass.Variabs.gamma * np.sqrt(np.abs(delta_p)))
        elif BigClass.Variabs.R_update == 'deltaR_propto_Q':  # delta_R propto flow Q
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * self.u)
        elif BigClass.Variabs.R_update == 'R_propto_Q':  # R propto flow Q
            self.R_in_t.append(BigClass.Variabs.gamma * self.u)
        elif BigClass.Variabs.R_update == 'R_propto_Q_exp':  # R propto flow Q
            R_max = copy.copy(BigClass.Variabs.R_max)
            R_min = copy.copy(BigClass.Variabs.R_min)
            R_bar: float = (R_max + R_min)/2.0
            u_0: float = 1 / (np.sqrt(BigClass.Strctr.NE) * R_bar)
            # R_nxt: float = R_max + (R_min - R_max) * np.exp(- self.u / u_0)
            R_nxt = R_max + (R_min - R_max) * np.exp(- np.abs(self.u) / u_0)
            self.R_in_t.append(BigClass.Variabs.gamma * R_nxt)
        elif BigClass.Variabs.R_update in {'deltaR_propto_dp_nonlin', 'deltaR_propto_dp_nonlin_decay'}:  # non linear rules 
            delta_R = BigClass.Variabs.gamma*(delta_p)**3 * update_cond
            if BigClass.Variabs.normalize_step:
                delta_R_norm = self.alpha * delta_R / np.linalg.norm(delta_R)
                R_nxt = self.R_in_t[-1] + delta_R_norm
            else:
                if BigClass.Variabs.R_update == 'deltaR_propto_dp_nonlin_decay':  # update and add decay of resistance to 1
                    R_nxt = self.R_in_t[-1] + delta_R - BigClass.Variabs.decay*(self.R_in_t[-1] - 1)
                else:
                    R_nxt = self.R_in_t[-1] + delta_R
            self.R_in_t.append(np.clip(R_nxt, 1e-12, None))
        elif BigClass.Variabs.R_update == 'grad_desc':
            if delta_K == []:
                print('error, no delta_K vector supplied')
            else:
                K_vec = matrix_functions.K_from_R(self.R_in_t[-1])
                K_vec_nxt = K_vec + self.alpha * delta_K
                R_nxt = 1/K_vec_nxt
                if BigClass.Variabs.normalize_step:
                    delta_R = R_nxt - self.R_in_t[-1]
                    delta_R_norm = self.alpha * delta_R / np.linalg.norm(delta_R)
                    R_nxt = self.R_in_t[-1] + delta_R_norm
            self.R_in_t.append(R_nxt)
        elif BigClass.Variabs.R_update == 'deltaR_propto_Power':  # delta_R propto Power dissipation dp*Q
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * self.u * delta_p * np.sign(delta_p))
        elif BigClass.Variabs.R_update == 'R_propto_Power':  # delta_R propto Power dissipation dp*Q
            self.R_in_t.append(BigClass.Variabs.gamma * self.u * delta_p * np.sign(delta_p))
        elif BigClass.Strctr.net_type == 'beads':
            self.R_in_t.append(matrix_functions.ChangeRFromFlow(BigClass, BigClass.Variabs.R_max,
                                                                BigClass.Variabs.R_min,
                                                                R_change_scheme='beads_pressure', allowed_cells=[],
                                                                beta=0.0))

        if BigClass.Variabs.hysteresis:
            BigClass.State.hysteresis[BigClass.State.hysteresis > BigClass.Variabs.hyst_thresh] = BigClass.Variabs.hyst_thresh
            BigClass.State.hysteresis[BigClass.State.hysteresis < -BigClass.Variabs.hyst_thresh] = -BigClass.Variabs.hyst_thresh

        # print
        if not BigClass.Variabs.supress_prints:
            pass

        self.R_in_t[-1][self.R_in_t[-1] < 10**-12] = 10**-12  # inhibit vanishing R - already accounted for above?

    def dK_grad_desc(self, Strctr: "Network_Structure", dK_step, p_desired, func):
        """
        Add desc

        inputs:
        BigClass  - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET)
                    and network state (State) class instances
                    I will not go into everything used from there to save space here.
        CstrTuple - Tuple consisting - Cstr_full - 2D array without last column, which is f from Rocks & Katifori 2018
                                                   https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
                                       Cstr -      Cstr_full without last line
                                       f    -      constraint vector (Rocks and Katifori 2018) 1D np.arrays [NEdges]
                                                   such that EI[i] is node connected to EJ[i] at certain edge
        K_vec     - 1D np.array [NE] of conductivities (inverse of resistances)
        p_desired - 1D np.array [Nout] of desired outputs given the inputs
        func      - str

        outputs:
        cost: np.float, MSE or mean abs between maesured and desired outputs
        """
        GD_dcost_vec = np.zeros([np.size(self.K_vec)])
        for m in range(np.size(self.K_vec)):
            dK_vec = np.zeros([np.size(self.K_vec)])
            dK_vec[m] = dK_step
            GD_dcost_vec[m] = self.calc_GD_cost(Strctr, p_desired, K_vec_for_GD=self.K_vec+dK_vec,
                                                mod='for_grad_desc', func=func)
            dcost_dK = (GD_dcost_vec - self.GD_cost) / dK_step
            delta_K = -dcost_dK
        return delta_K

    def calc_GD_cost(self, Strctr: "Network_Structure", p_desired: NDArray[np.float_], K_vec_for_GD=[],
                     mod: str = 'measure', func: str = 'MSE') -> np.float_:
        """
        MSE or mean abs cost between desired and measured network output
        given pressure input, conductivities and constraint matrix

        inputs:
        BigClass  - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET)
                    and network state (State) class instances
                    I will not go into everything used from there to save space here.
        CstrTuple - Tuple consisting - Cstr_full - 2D array without last column, which is f from Rocks & Katifori 2018
                                                   https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
                                       Cstr -      Cstr_full without last line
                                       f    -      constraint vector (Rocks and Katifori 2018) 1D np.arrays sized NEdges
                                                   such that EI[i] is node connected to EJ[i] at certain edge
        K_vec     - 1D np.array [NE] of conductivities (inverse of resistances)
        p_desired - 1D np.array [Nout] of desired outputs given the inputs

        outputs:
        cost: np.float, MSE  or mean abs between maesured and desired outputs
        """
        if K_vec_for_GD == []:
            K_vec = self.K_vec
        else:
            K_vec = K_vec_for_GD
        p, u = solve.solve_flow(Strctr, self.CstrTuple, K_vec)
        p_out: NDArray[np.float_] = p[Strctr.output_nodes_arr][:, 0]  # p at output nodes, indexed as 1D array

        if p_out.size == p_desired.size:
            if func == 'MSE':
                cost: np.float_ = np.mean((p_out - p_desired) ** 2)
            elif func == 'mean_abs':
                cost = np.mean(np.abs(p_out - p_desired))
        else:
            cost = np.nan
            print(f"Incompatible sizes, p_out shape = {p_out.shape}, p_desired shape = {p_desired.shape}")
        if mod == 'measure':  # save in State class only if not part of dK_grad_desc
            self.GD_cost: np.float_ = cost
        return cost

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
        self.loss_in_t.append(self.loss)

    def calc_update_vals_vec(self, BigClass: "Big_Class") -> None:
        """
        calculate the update modality values of inputs and outputs if the scheme is Adaline-like (standard way)
        or GD_like (not used)
        Adaline-like - delta_p_{ij}^!= alpha/gamma * (y_i-x_j) * (y_hat_i-y_i) as in the paper.
        GD-like      - delta_p_{ij}^!= -alpha*(Loss-Loss')/(y_i-x_j-y_i'+x_j')
                       In both cases you then multiply by U^dagger to make it realizeable on BEASTS

        input:
        BigClass

        output:
        update_vec - values for "update" modality pressures, nodes numbered as in Strctr.DM
        """
        in_nodes = copy.copy(BigClass.Strctr.input_nodes_arr)
        out_nodes = copy.copy(BigClass.Strctr.output_nodes_arr)
        ground_nodes = copy.copy(BigClass.Strctr.ground_nodes_arr)
        if BigClass.Variabs.training_scheme == 'GD_like':
            L_vec: NDArray[np.float_] = np.zeros(BigClass.Strctr.NN)
            L_vec[out_nodes] = self.loss
            delta_p = np.matmul(BigClass.Strctr.DM, self.p[:BigClass.Strctr.NN]).T
            delta_p[delta_p == 0] = 10**(-9)  # correct for division by zero
            one_over_delta_p_norm = 1 / delta_p / np.linalg.norm(1 / delta_p)  # normalize division by pressure diffs
            C_vec: NDArray[np.float_] = np.matmul(BigClass.Strctr.RM, L_vec) * one_over_delta_p_norm
            # C_vec: NDArray[np.float_] = np.matmul(Strctr.RM, L_vec) / delta_p
            if BigClass.Variabs.R_update in ['deltaR_propto_dp_nonlin', 'deltaR_propto_dp_nonlin_decay']:  # normalize C as well
                C_vec_norm = C_vec[0] / np.linalg.norm(C_vec[0])
                update_vec: NDArray[np.float_] = - self.alpha * np.matmul(BigClass.Strctr.DM_dagger, C_vec_norm)
            else:
                update_vec = - self.alpha * np.matmul(BigClass.Strctr.DM_dagger, C_vec[0])
        elif BigClass.Variabs.training_scheme == 'Adaline':
            if BigClass.Variabs.Ninter > 0:
                Strctr = BigClass.Strctr_fict
            else:
                Strctr = BigClass.Strctr
            p = np.concatenate([BigClass.State.p[in_nodes], BigClass.State.p[out_nodes],
                                BigClass.State.p[ground_nodes]])
            grad_loss_vec = matrix_functions.grad_loss_FC(Strctr.NE, p, Strctr.DM, Strctr.output_nodes_arr,
                                                          Strctr.ground_nodes_arr, BigClass.State.loss)
            self.grad_loss_vec = grad_loss_vec
            grad_loss_vec_norm = grad_loss_vec / np.linalg.norm(grad_loss_vec)
            self.grad_loss_vec_norm = grad_loss_vec_norm
            if BigClass.Variabs.normalize_loss:  # normalize C as well
                update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec_norm)
            else:
                update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec)
            if BigClass.Variabs.Ninter > 0:  # enlarge update_vec again for complying with Strctr
                # Insert zeros at each index, shifting elements to the right
                for idx in BigClass.Strctr.inter_nodes_arr:
                    update_vec = np.insert(update_vec, idx, 0)
            # update_vec[-1] = 0  # neglect ground node
            # grad_loss_vec[-1] = 0
        self.update_vec = update_vec
        self.update_vec_in_t.append(update_vec)

    def calc_Power_norm(self, BigClass: "Big_Class"):
        self.Power_norm = statistics.power_dissip_norm(self.u, self.R_in_t[-1], self.input_drawn)
        self.Power_norm_in_t.append(self.Power_norm)

        # print
        if not BigClass.Variabs.supress_prints:
            print('Power dissipation normalized', self.Power_norm)

    def calculate_accuracy_fullDataset(self, BigClass: "Big_Class") -> None:
        self.accuracy_vec: NDArray[np.int_] = zeros(np.shape(BigClass.Variabs.dataset)[0], dtype=np.int_)
        for i, datapoint in enumerate(BigClass.Variabs.dataset):
            self.draw_p_in_and_desired(BigClass.Variabs, i, modality='measure_for_accuracy')
            self.solve_flow_given_modality(BigClass, "measure_for_accuracy")  # measure and don't change resistances
            self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
                                                                         BigClass.Variabs.targets[i])
        self.accuracy = np.mean(self.accuracy_vec)

    def calculate_accuracy_testset(self, BigClass: "Big_Class") -> None:
        self.accuracy_vec = zeros(np.shape(BigClass.Variabs.X_test)[0], dtype=np.int_)
        for i, datapoint in enumerate(BigClass.Variabs.X_test):
            self.draw_p_in_and_desired(BigClass.Variabs, i, modality='measure_for_accuracy')
            self.solve_flow_given_modality(BigClass, "measure_for_accuracy")  # measure and don't change resistances
            self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
                                                                         BigClass.Variabs.y_test[i])
        self.accuracy = np.mean(self.accuracy_vec)

    def measure_flow(self, BigClass: "Big_Class") -> None:
        """
        Measure flow u (or Q) from all output nodes of network; summing on flow over all edges connected to output nodes

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        u_out: flow from all output nodes np.ndarray sized [Nout,]
        """
        self.u_out = np.sum(self.u[BigClass.Strctr.output_edges]*BigClass.Strctr.output_edge_directions)
