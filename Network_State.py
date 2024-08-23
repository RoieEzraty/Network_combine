from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array, zeros, arange
from typing import TYPE_CHECKING, Callable, Union

import matrix_functions, functions, solve, plot_functions

if TYPE_CHECKING:
    from Network_Structure import Network_Structure
    from User_Variables import User_Variables
    from Big_Class import Big_Class


############# Class - network state variables #############


class Network_State():
    """
    Class with variables that hold information of state of network.
    what ends with _in_t holds all time instances of the variable, each list index is different t
    what ends w/out _in_t is at current time instance self.t
    """
    def __init__(self, Nin: int, Nout: int) -> None:
        super(Network_State, self).__init__()  
        self.t: int = 0  
        self.p: np.ndarray=array([])  
        self.u: np.ndarray=array([])  
        self.output: np.ndarray=array([])
        self.output_in_t: List[np.ndarray] = []
        self.desired: np.ndarray=array([])
        self.input_drawn_in_t: List[np.ndarray] = []
        self.desired_in_t: List[np.ndarray] = []
        self.output_dual_in_t: List[np.ndarray] = [0.5*np.ones(Nout)]
        self.input_dual_in_t: List[np.ndarray] = [1.0*np.ones(Nin)]
        self.loss_in_t: List[np.ndarray] = []   
            
    def initiate_resistances(self, BigClass: "Big_Class") -> None:
        """
        After using build_incidence, initiate resistances
        """
        self.R_in_t: List[np.ndarray] = [np.ones((BigClass.Strctr.NE), dtype=float)]
            
    def draw_p_in_and_desired(self, Variabs: "User_Variables"):
        """
        Every time step, draw random input pressures and calculate the desired output given input

        inputs:
        Variabs: User_Variables class

        outputs
        p_drawn: np.ndarray sized [Nout,], input pressures
        desired: np.ndarray sized [Nout,], desired output defined by the task M*p_input
        """
        self.input_drawn: np.ndarray = np.random.uniform(low=0.0, high=2.0, size=Variabs.Nin)
        self.desired = np.matmul(Variabs.M, self.input_drawn)
        self.input_drawn_in_t.append(self.input_drawn)
        self.desired_in_t.append(self.desired)

    def assign_output_dual(self, Variabs: "User_Variables"):
        """
        add desc
        """
        raise NotImplementedError('still need to decide how to assign output dual')
        self.output_dual: np.ndarray = zeros(Variabs.Nout)
        self.output_dual_in_t.append(self.output_dual)

    def solve_flow_until_conv(self, BigClass: "Big_Class", problem: str) -> None:
        """
        solve_flow_until_conv solves the flow under same BCs while updating K until convergence. 
        used as part of flow_iterate()
        uses solve_flow_const_K()

        inputs:
        BigClass -       class instance including user variables (Variabs), network structure (Strctr) and networkx (NET) and network state (State) class instances
                         I will not go into everything used from there to save space here.
        u              - 1D array sized [NE + constraints, ], flow at each edge at beginning of iteration
        Cstr           - 2D array without last column (which is f from Rocks and Katifori 2018)
        f              - constraint vector (from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
        iters_same_BCs - int, maximal # iterations under same boundary condition
        sim_type       - str, simulation type, see flow_iterate() function for descrip.

        outputs:
        p     - pressure at every node under the specific BC, after convergence while allowing conductivities to change
        u_nxt - flow at every edge under the specific BC, after convergence while allowing conductivities to change
        """
        if problem == 'measure':
            CstrTuple = functions.setup_constraints_given_pin((BigClass.Strctr.input_nodes_arr, BigClass.Strctr.ground_nodes_arr), BigClass.State.input_drawn, BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
            self.p, self.u = solve.solve_flow_const_K(BigClass, CstrTuple, self.R_in_t[-1])
            self.output = self.p[BigClass.Strctr.output_nodes_arr].ravel()  # output is only at output edges, raveled so sized [Nout,]
            self.output_in_t.append(self.output)
        elif problem == 'dual':
            CstrTuple = functions.setup_constraints_given_pin((BigClass.Strctr.input_nodes_arr, BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.output_nodes_arr), (BigClass.State.input_dual_in_t[-1], BigClass.State.output_dual_in_t[-1]),\
                                                              BigClass.Strctr.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ)
            self.p, self.u = solve.solve_flow_const_K(BigClass, CstrTuple, self.R_in_t[-1])
        print('Rs', self.R_in_t[-1])
        

    def calc_loss(self, BigClass: "Big_Class") -> None:
        """
        Calculates the loss given system state and desired outputs, perhaps including 1 time step ago

        inputs:
        BigClass: Class instance where all the

        outputs:
        loss: np.ndarray sized [Nout,]
        """
        if BigClass.Variabs.loss_fn == functions.loss_fn_2samples:
            self.loss = BigClass.Variabs.loss_fn(self.output, self.output_in_t[-2], self.desired, self.desired_in_t[-2])
        elif BigClass.Variabs.loss_fn == functions.loss_fn_1sample:
            self.loss = BigClass.Variabs.loss_fn(self.output, self.desired)
        self.loss_in_t.append(self.loss)

    def update_pressure_dual(self, BigClass: "Big_Class") -> None:
        self.t += 1  # update time
        loss = self.loss_in_t[-1]  # copy loss
        input_dual = self.input_dual_in_t[-1]
        pert = np.random.normal(size=np.size(input_dual))  # perturbation, not in use
        input_drawn = self.input_drawn_in_t[-1]
        # dot product for alpha in pressure update
        if BigClass.Variabs.use_p_tag:
            input_drawn_prev = self.input_drawn_in_t[-2]
            print('delta_loss', loss[0]-loss[1])
            print('delta_input', input_drawn-input_drawn_prev)
            print('the dot itself', (input_drawn-input_drawn_prev)*np.dot(BigClass.Variabs.alpha_vec, loss[0]-loss[1]))
            self.input_dual_nxt = input_dual - (input_drawn-input_drawn_prev)*np.dot(BigClass.Variabs.alpha_vec, loss[0]-loss[1])
            print('input_dual_nxt for inside function', self.input_dual_nxt)
        else:
            self.input_dual_nxt = input_dual - (input_drawn)*np.dot(BigClass.Variabs.alpha_vec, loss[0])                     
        if BigClass.Variabs.supress_prints:
            pass
        else:
            print('loss=', loss)
            print('time=', self.t)
            print('input_dual_nxt=', self.input_dual_nxt)

        # if pressure changes without memory
        if BigClass.Variabs.R_update == 'deltaR' and np.shape(self.input_dual_in_t)[0]>1:  # make sure its not initial value
            self.input_dual_nxt -= input_dual  # erase memory

        self.input_dual_in_t.append(self.input_dual_nxt)  # append into list in time

    def update_output_dual(self, BigClass: "Big_Class"):
        loss = self.loss_in_t[-1]
        pert = np.random.normal(size=np.size(self.output))
        output_dual = copy.copy(self.output_dual_in_t[-1])
        # element-wise multiplication for alpha in output update
        # self.output = out_dual + self.variabs.alpha * np.dot(self.output-self.out_in_t[-2], loss[0]-loss[1])
        print('x-xprime', self.output-self.output_in_t[-2])
        print('loss-loss_prime', loss[0]-loss[1])
        self.output_dual_nxt = output_dual + BigClass.Variabs.alpha_vec * (self.output-self.output_in_t[-2])*(loss[0]-loss[1])
        # self.output = out_dual + self.variabs.alpha[1] * (self.output[1]-self.out_in_t[-2][1])*(loss[0][1]-loss[1][1])
        
        # outputs change without memory?
        if BigClass.Variabs.R_update == 'deltaR' and np.shape(self.output_dual_in_t)[0]>1:  # make sure its not initial value
            self.output_dual_nxt -= output_dual  # erase memory
        self.output_dual_in_t.append(self.output_dual_nxt)           
        
        # optionally print output
        if BigClass.Variabs.supress_prints:
            pass
        else:
            print('output_dual_nxt', self.output_dual_nxt)

    def update_Rs(self, BigClass: "Big_Class") -> None:
        R_vec: np.ndarray = self.R_in_t[-1]
        delta_p: np.ndarray = self.u * R_vec
        if BigClass.Variabs.R_update == 'deltaR':
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * delta_p)
        elif BigClass.Variabs.R_update == 'propto':
            self.R_in_t.append(BigClass.Variabs.gamma * delta_p)
          
        # optionally display resistances
        if BigClass.Variabs.supress_prints:
            pass
        else:
            print('R_nxt', self.R_in_t[-1])