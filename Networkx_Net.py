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

        # in DM columns are nodes so node=i
        # Reorder p according to NET.nodes
        self.p_reordered = array([p[node] for node in self.NET.nodes])
