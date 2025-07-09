from __future__ import annotations
import networkx as nx
import numpy as np

from typing import Tuple, List, Dict, Any
from numpy import array, zeros
from typing import TYPE_CHECKING
from numpy.typing import NDArray

import plot_functions

if TYPE_CHECKING:
    from Big_Class import Big_Class


# ===================================================
# Class - User Variables
# ===================================================


class Networkx_Net:
    """
    Networkx_net contains networkx data for plots
    """
    def __init__(self, scale: float, squish: float) -> None:
        super(Networkx_Net, self).__init__()
        self.scale = scale
        self.squish = squish

    def buildNetwork(self, BigClass: "Big_Class") -> None:
        """
        Builds a networkx network using edges from EIEJ_plots which are built upon EI and EJ at "Matrixfuncs.py"
        After this step, the order of edges at EIEJ_plots and in the networkx net is not the same which is shit

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.

        outputs:
        NET - networkx network containing just the edges from EIEJ_plots
        """
        NET: nx.DiGraph = nx.DiGraph()  # initiate graph object
        NET.add_edges_from(BigClass.Strctr.EIEJ_plots)  # add edges
        self.NET: nx.DiGraph = NET

    def build_pos_lattice(self, BigClass: "Big_Class", plot: bool = False,
                          node_labels: bool = False) -> None:
        """
        build_pos_lattice builds the lattice of positions of edges and nodes

        inputs:
        BigClass    - class instance including User_Variables, Network_Structure instances, etc.
        plot        - bool, whether to plot or not
        node_labels - boolean, show node number in plot or not

        outputs:
        pos_lattice - dict, positions of nodes from NET.nodes
        """
        if BigClass.Strctr.net_type == 'square':
            height = BigClass.Strctr.net_height
            pos_lattice: Dict[Any, Tuple[float, float]] = \
                {index: (index % height, index // height) for index in range(len(self.NET.nodes))}
        elif BigClass.Strctr.net_type == 'beads':
            pos_lattice = {}
            num_cols = BigClass.Strctr.net_len
            num_rows = BigClass.Strctr.net_height
            for i in range(num_rows):
                for j in range(num_cols):
                    # Compute a flat index for each cross
                    cross_index = i * num_cols + j
                    start_index = cross_index * 5

                    # Define the x and y offsets for this cross
                    x_offset = self.scale * j
                    y_offset = self.scale * i

                    # Assign positions for the 5 nodes
                    # (left, lower, right, upper, middle).
                    pos_lattice[start_index + 0] = array([-(self.scale / 2 - self.squish) + x_offset, 0 + y_offset])
                    pos_lattice[start_index + 1] = array([0 + x_offset, -(self.scale / 2 - self.squish) + y_offset])
                    pos_lattice[start_index + 2] = array([(self.scale / 2 - self.squish) + x_offset, 0 + y_offset])
                    pos_lattice[start_index + 3] = array([0 + x_offset, (self.scale / 2 - self.squish) + y_offset])
                    pos_lattice[start_index + 4] = array([0 + x_offset, 0 + y_offset])
        elif BigClass.Strctr.net_type == 'FC':
            pos_lattice = {}

            k = 0  # horizontal position

            # Input layer: x = 0
            n_in = len(BigClass.Strctr.input_nodes_arr)
            for i, node in enumerate(BigClass.Strctr.input_nodes_arr):
                y = i - (n_in - 1) / 2  # Center the input layer around x=0
                pos_lattice[node] = (k, y)
            k += 1

            n_inter = len(BigClass.Strctr.inter_nodes_arr)
            for i, node in enumerate(BigClass.Strctr.inter_nodes_arr):
                y = i - (n_inter - 1) / 2  # Center the input layer around x=0
                pos_lattice[node] = (k, y)
            k += 1

            # Output layer: x = 1
            n_out = len(BigClass.Strctr.output_nodes_arr)
            for i, node in enumerate(BigClass.Strctr.output_nodes_arr):
                y = i - (n_out - 1) / 2  # Center the output layer around x=0
                pos_lattice[node] = (k, y)
            k += 1

            # Ground node: x = 2, centered
            pos_lattice[BigClass.Strctr.NN-1] = (k, 0)

        else:
            pos_lattice = nx.spring_layout(self.NET, k=1.0, iterations=20)
        self.pos_lattice = pos_lattice
        if plot:
            plot_functions.plotNetStructure(self.NET, BigClass, pos_lattice, node_labels=node_labels)

    def save_R_reordered(self, R_vec: NDArray[np.float_], EIEJ_plots: list[Tuple]) -> None:

        # Create a mapping from edges to their index in Strctr.EIEJ_plots
        edge_to_index = {edge: idx for idx, edge in enumerate(EIEJ_plots)}

        # Reorder R_in_t according to NET.NET.edges
        self.R_reordered = array([R_vec[edge_to_index[edge]] for edge in self.NET.edges])

    def save_u_reordered(self, u: NDArray[np.float_], EIEJ_plots: list[Tuple]) -> None:

        # Create a mapping from edges to their index in Strctr.EIEJ_plots
        edge_to_index = {edge: idx for idx, edge in enumerate(EIEJ_plots)}

        # Reorder R_in_t according to NET.NET.edges
        self.u_reordered = array([u[edge_to_index[edge]] for edge in self.NET.edges])

    def save_p_reordered(self, p: NDArray[np.float_]) -> None:
        # Reorder p according to NET.nodes in DM columns are nodes so node=i
        self.p_reordered = array([p[node] for node in self.NET.nodes])
