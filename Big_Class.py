from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from User_Variables import User_Variables
	from Network_State import Network_State
	from Network_Structure import Network_Structure


############# Class - Big class that contains all smaller classes #############


class Big_Class:
	"""
	Big_Class contains the main classes under Network Simulation
	"""
	
	def __init__(self, Variabs: "User_Variables"):
		self.Variabs = Variabs

	def add_Strctr(self, Strctr: "Network_Structure"):
		self.Strctr = Strctr

	def add_State(self, State: "Network_State"):
		self.State = State

	def add_NET(self, NET):
		self.NET = NET
