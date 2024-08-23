from __future__ import annotations
import numpy as np
import random
import copy
import networkx as nx

from typing import Tuple, List
from numpy import array as array
from numpy import zeros as zeros
from typing import TYPE_CHECKING

import plot_functions

if TYPE_CHECKING:
    from Big_Class import Big_Class


############# Class - User Variables #############


class Networkx_Net:
	"""
	Networkx_net contains networkx data for plots

	inputs:

	outputs:
	NET - networkx net object (initially empty)
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
	    EIEJ_plots - 2D array sized [NE, 2] - 
	                 EIEJ_plots[i,0] and EIEJ_plots[i,1] are input and output nodes to edge i, respectively
	    
	    outputs:
	    NET - networkx network containing just the edges from EIEJ_plots
	    """
		NET: nx.DiGraph = nx.DiGraph()  # initiate graph object
		NET.add_edges_from(BigClass.Strctr.EIEJ_plots)  # add edges 
		self.NET: nx.DiGraph = NET

	def build_pos_lattice(self, BigClass: "Big_Class", plot: bool=False, node_labels: bool=False):
		"""
		build_pos_lattice builds the lattice of positions of edges and nodes

		inputs:
		BigClass - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET) and network state (State) class instances

	    outputs:
	    pos_lattice - dict, positions of nodes from NET.nodes
		"""
		self.pos_lattice = plot_functions.plotNetStructure(self.NET, plot=plot, node_labels=node_labels)