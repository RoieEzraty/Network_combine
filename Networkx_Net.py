from __future__ import annotations
import networkx as nx

from typing import Tuple, List, Dict, Any
from numpy import array, zeros
from typing import TYPE_CHECKING

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

    def build_pos_lattice(self, BigClass: "Big_Class", plot: bool = False, node_labels: bool = False) -> None:
        """
        build_pos_lattice builds the lattice of positions of edges and nodes

        inputs:
        BigClass    - class instance including User_Variables, Network_Structure instances, etc.
        plot        - bool, whether to plot or not
        node_labels - boolean, show node number in plot or not

        outputs:
        pos_lattice - dict, positions of nodes from NET.nodes
        """
        pos_lattice: Dict[Any, Tuple[float, float]] = plot_functions.plotNetStructure(self.NET, plot=plot,
                                                                                      node_labels=node_labels)
        self.pos_lattice = pos_lattice
