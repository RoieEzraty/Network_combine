from __future__ import annotations
import numpy as np
import random
import copy

from typing import Tuple, List
from numpy import array, zeros, arange
from typing import TYPE_CHECKING

import matrix_functions, functions, solve

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
        self.out_in_t: List[np.ndarray] = []
        self.loss_in_t: List[np.ndarray] = []
        self.t: int = 0
        self.out_dual_in_t: List[np.ndarray] = [0.5*np.ones(Nout)]
        self.p_dual_in_t: List[np.ndarray] = [1.0*np.ones(Nin)]
        self.p_drawn_in_t: List[np.ndarray] = []
        self.desired_in_t: List[np.ndarray] = []
        self.output_dual_in_t: List[np.ndarray] = []
            
    def initiate_resistances(self, Strctr: "Network_Structure") -> None:
        """
        After using build_incidence, initiate resistances
        """
        self.R_in_t: List[np.ndarray] = [np.ones((Strctr.NE), dtype=float)]
            
    def draw_p_in_and_desired(self, Variabs: "User_Variables"):
        """
        Every time step, draw random input pressures and calculate the desired output given input

        inputs:
        Variabs: User_Variables class

        outputs
        p_drawn: np.ndarray sized [Nout,], input pressures
        desired: np.ndarray sized [Nout,], desired output defined by the task M*p_input
        """
        self.p_drawn: np.ndarray = np.random.uniform(low=0.0, high=2.0, size=Variabs.Nin)
        self.desired: np.ndarray = np.matmul(Variabs.M, self.p_drawn)
        self.p_drawn_in_t.append(self.p_drawn)
        self.desired_in_t.append(self.desired)

    def assign_output_dual(self, Variabs: "User_Variables"):
        """
        add desc
        """
        raise NotImplementedError('still need to decide how to assign output dual')
        self.output_dual: np.ndarray = zeros(Variabs.Nout)
        self.output_dual_in_t.append(self.output_dual)

    def solve_flow_until_conv(self, BigClass: "Big_Class", CstrTuple: Tuple[np.ndarray, np.ndarray, np.ndarray]):
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
        p, u_nxt = self.solve_flow_const_K(BigClass, CstrTuple, self.R_in_t[-1])
        return p, u_nxt

    def solve_flow_const_K(self, BigClass: "Big_Class", CstrTuple: Tuple[np.ndarray, np.ndarray, np.ndarray], R_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        K_mat = np.eye(BigClass.Strctr.NE) / R_vec
        L, L_bar = matrix_functions.buildL(BigClass, BigClass.Strctr.DM, K_mat, Cstr, BigClass.Strctr.NN)  # Lagrangian
        p, u_nxt = solve.Solve_flow(L_bar, BigClass.Strctr.EI, BigClass.Strctr.EJ, K_eff, f, round=10**-10)  # pressure and flow
        # NETfuncs.PlotNetwork(p, u_nxt, self.K, BigClass, BigClass.Strctr.EIEJ_plots, BigClass.Strctr.NN, 
        #                    BigClass.Strctr.NE, nodes='yes', edges='yes', savefig='no')
        return p, u_nxt

    def setup_constraints_given_pin(self, BigClass: "Big_Class", problem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        build_incidence builds the incidence matrix DM

        inputs:
        BigClass: class instance consisting User_Variables, Network_State, etc.
        problem: str, "measure" for pressure posed on inputs and ground, "dual" for pressure also on outputs and change of resistances 

        outputs:
        Cstr_full = 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges. last column is value of constraint
                    (p value of row contains just +1. pressure drop if row contains +1 and -1)
        Cstr      = 2D array without last column (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
        f         = constraint vector (from Rocks and Katifori 2018)
        """
        # specific constraints for training step 
        NodeData, Nodes, GroundNodes = self.Constraints_nodes(BigClass, problem)

        print('NodeData', NodeData)
        print('Nodes', Nodes)
        print('GroundNodes',  GroundNodes)

        # BC and constraints as matrix
        Cstr_full, Cstr, f = self.ConstraintMatrix(NodeData, Nodes, GroundNodes, BigClass.Variabs.NN, BigClass.Strctr.EI, BigClass.Strctr.EJ) 
        return Cstr_full, Cstr, f 

    def Constraints_nodes(self, BigClass: "Big_Class", problem: str):
        """
        Constraints_afo_task sets up the constraints on nodes and edges for specific learning task, and for specific step.
        This comes after Setup_constraints which sets them for the whole task

        inputs:
        BigClass: class instance that includes all class instances User_Variables, Network_State, et.c
        problem: str, "measure" for pressure posed on inputs and ground, "dual" for pressure also on outputs and change of resistances  
        """
        GroundNodes: np.ndarray = copy.copy(BigClass.Strctr.ground_nodes_arr)
        InNodeData: np.ndarray = copy.copy(BigClass.State.p_drawn)
        InNodes: np.ndarray = copy.copy(BigClass.Strctr.input_nodes_arr)
        print('BigClass input nodes', BigClass.Strctr.input_nodes_arr)
        print('InNodes', InNodes)
        if problem=="measure":
            OutputNodeData: np.ndarray = array([], dtype=int)
            OutputNodes: np.ndarray = array([], dtype=int)
        elif problem=="dual":
            OutputNodeData = copy.copy(BigClass.State.output_dual)
            OutputNodes = copy.copy(BigClass.Strctr.output_nodes_arr)
        NodeData: np.ndarray = np.append(InNodeData, OutputNodeData)
        Nodes: np.ndarray = np.append(InNodes, OutputNodes)
        print('Nodes inside func', Nodes)
        return NodeData, Nodes, GroundNodes

    def ConstraintMatrix(self, NodeData, Nodes, GroundNodes, NN, EI, EJ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds constraint matrix, 
        for constraints on edge voltage drops: 1 at input node index, -1 at output and voltage drop at NN+1 index, for every row
        For constraints on node voltages: 1 at constrained node index, voltage at NN+1 index, for every row
        For ground nodes: 1 at ground node index, 0 else.

        Inputs:
        NodeData    = 1D array at length as "Nodes" corresponding to pressures at each node from "Nodes"
        Nodes       = 1D array of nodes that have a constraint
        GroundNodes = 1D array of nodes that have a constraint of ground (outlet)
        NN          = int, number of nodes in network
        EI          = 1D array of nodes at each edge beginning
        EJ          = 1D array of nodes at each edge ending corresponding to EI

        outputs:
        Cstr_full = 2D array sized [Constraints, NN + 1] representing constraints on nodes and edges. last column is value of constraint
                 (p value of row contains just +1. pressure drop if row contains +1 and -1)
        Cstr      = 2D array without last column (which is f from Rocks and Katifori 2018 https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116)
        f         = constraint vector (from Rocks and Katifori 2018)
        """

        # ground nodes
        csg = len(GroundNodes)
        idg = arange(csg)
        CStr = zeros([csg, NN+1])
        CStr[idg, GroundNodes] = +1.
        CStr[:, NN] = 0.
        
        # constrained node pressures
        if len(Nodes):
            csn = len(Nodes)
            idn = arange(csn)
            SN = zeros([csn, NN+1])
            SN[idn, Nodes] = +1.
            SN[:, NN] = NodeData
            CStr = np.r_[CStr, SN]

        # to not lose functionality in the future if I want to add Edges as well
        Edges = array([])
        EdgeData = array([])
        
        # constrained edge pressure drops
        if len(Edges):
            cse = len(Edges)
            ide = arange(cse)
            SE = zeros([cse, NN+1])
            SE[ide, EI[Edges]] = +1.
            SE[ide, EJ[Edges]] = -1.
            SE[:, NN] = EdgeData
            CStr = np.r_[CStr, SE]
            
        # last column of CStr is vector f 
        f = zeros([NN + len(CStr), 1])
        f[NN:,0] = CStr[:,-1]

        return CStr, CStr[:,:-1], f
