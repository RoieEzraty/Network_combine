from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Any
from typing import TYPE_CHECKING
from numpy.typing import NDArray

from typing import Tuple, List, Union, Optional

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Network_State import Network_State


# ================================
# functions for plots
# ================================


def plot_importants(State: "Network_State", Variabs: "User_Variables", desired: List[NDArray[np.float_]],
                    M: Optional[NDArray[np.int_]] = None, include_network: Optional[bool] = False,
                    NET: Optional[nx.DiGraph] = None) -> None:
    """
    one plot with 4 subfigures of
    1) output / desired - 1.
    2) inputs and outputs of the dual problem
    3) resistances in time
    4) absolute mean value of loss in time

    inputs:
    State   - class instance of the state variables of network
    Variabs - class instance of the variables by the user
    desired - List of arrays of desired outputs given the drawn inputs and task matrix M
    M       - task matrix M under which desired output = M*input

    outputs:
    1 matplotlib plot
    """
    if Variabs.Nin == 1 and Variabs.Nout == 1:  # 1by1, simplest
        if M is not None:
            A: float = M[0]
            R_theor: NDArray[np.float_] = np.array([(1-A)/A])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
    elif Variabs.Nin == 1 and Variabs.Nout == 2:  # Allostery
        if M is not None:
            A = M[0]  # A = x_hat/p_in
            B: float = M[1]  # B = y_hat/p_in
            Rl_subs: float = 1.0
            R_theor = State.input_drawn_in_t[0]*np.array([(1-A)/(A*(1+1/Rl_subs)-B/Rl_subs),
                                                          (1-B)/(B*(1+1/Rl_subs)-A/Rl_subs)])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p\,\mathrm{dual}$']
        # legend3 = [r'$R_1$', r'$R_2$', r'$R_1\,\mathrm{theoretical}$', r'$R_2\,\mathrm{theoretical}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
    elif Variabs.Nin == 2 and Variabs.Nout == 1:  # Regression
        if M is not None:
            A = M[0, 0]
            B = M[0, 1]
            Rl_subs = 1
            R_theor = np.array([Rl_subs*(1-A-B)/A, Rl_subs*(1-A-B)/B])
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        # legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_1\,\mathrm{theoretical}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
    elif Variabs.Nin == 2 and Variabs.Nout == 3:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$z\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$',
                   r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$', r'$R_6$']
    elif Variabs.Nin == 2 and Variabs.Nout == 2:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$']
        if Variabs.access_interNodes:
            legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$',
                       r'$\mathrm{inter1\,dual}$', r'$\mathrm{inter2\,dual}$']
        else:
            legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$']
        legend3 = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$']
    elif Variabs.Nin == 3 and Variabs.Nout == 3:
        legend1 = [r'$\frac{x}{x\,\mathrm{desired}}$', r'$\frac{y}{y\,\mathrm{desired}}$',
                   r'$\frac{z}{z\,\mathrm{desired}}$']
        legend2 = [r'$x\,\mathrm{dual}$', r'$y\,\mathrm{dual}$', r'$z\,\mathrm{dual}$', r'$p_1\,\mathrm{dual}$',
                   r'$p_2\,\mathrm{dual}$', r'$p_3\,\mathrm{dual}$']
        legend3 = []
    elif Variabs.task_type == 'Iris_classification':
        legend1 = [r'$\mathrm{Setosa}$', r'$\mathrm{Verisicolor}$', r'$\mathrm{Virginica}$']
        legend2 = [r'$\mathrm{Setosa\,dual}$', r'$\mathrm{Verisicolor\,dual}$',
                   r'$\mathrm{Virginica\,dual}$', r'$p_1\,\mathrm{dual}$', r'$p_2\,\mathrm{dual}$',
                   r'$p_3\,\mathrm{dual}$', r'$p_4\,\mathrm{dual}$']
        legend3 = []
    else:
        legend1 = []
        legend2 = []
        legend3 = []
    legend4 = ['|loss|']
    if include_network:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 3))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.2))
    if Variabs.task_type != 'Iris_classification':
        ax1.plot(np.linspace(0, State.t, np.shape(State.output_in_t)[0]).T,
                 np.asarray(State.output_in_t)/np.asarray(State.desired_in_t)-1)
    else:
        ax1.plot(np.linspace(0, State.t, np.shape(State.output_in_t)[0]).T,
                 np.asarray(State.output_in_t))
    ax1.set_title('output in time')
    ax1.set_xlabel('t')
    if legend1:
        ax1.legend(legend1)
    ax2.plot(State.output_dual_in_t[1:])
    ax2.plot(State.input_dual_in_t[1:])
    if Variabs.access_interNodes:
        ax2.plot(State.inter_dual_in_t[1:])
    ax2.set_title('dual and p in time')
    ax2.set_xlabel('t')
    # ax2.set_ylim([-0.2,0.2])
    if legend2:
        ax2.legend(legend2)
    ax3.plot(State.R_in_t[1:])
    # if 'R_theor' in locals():  # if theoretical values were calculated, plot them
    #     print('R theoretical', R_theor)
    #     ax3.plot(np.outer(R_theor, np.ones(State.t)).T, '--')
    ax3.set_title('R in time')
    ax3.set_xlabel('t')
    if legend3:
        ax3.legend(legend3)
    for t in range(State.t):
        if t % len(Variabs.dataset) == 0 and t != 0:
            ax4.axvline(x=t, color='red', linestyle='--', linewidth=1)
    ax4.plot(np.mean(np.mean(np.abs(State.loss_norm_in_t[1:]), axis=1), axis=1))
    # ax4.plot(np.mean(np.mean(np.abs(State.loss_norm_in_t[1:]), axis=1), axis=1)/np.mean(np.abs(Variabs.targets), axis=1))
    ax4.set_xlabel('t')
    ax4.set_yscale('log')
    if legend4:
        ax4.legend(legend4)
    # fig.suptitle(f'alpha={Variabs.alpha_vec}')
    if include_network:
        if NET is not None:
            nx.draw_networkx(NET.NET, pos=NET.pos_lattice, edge_color='b', node_color='b', with_labels=True, ax=ax5)
        else:
            print('no NET assigned in input')
    plt.show()


def plotNetStructure(NET: nx.DiGraph, plot: bool = False, node_labels: bool = False) -> Dict[Any, Tuple[float, float]]:
    """
    Plots the structure (nodes and edges) of networkx NET

    input:
    NET         - networkx net of nodes and edges
    plot        - bool, whether to plot or not
    node_labels - boolean, show node number in plot or not

    output:
    pos_lattice - dict of positions of nodes from NET.nodes
    if plot=='yes' also show matplotlib plot of network structure
    """
    pos_lattice: Dict[Any, Tuple[float, float]] = nx.spring_layout(NET, k=1.0, iterations=20)
    if plot:
        nx.draw_networkx(NET, pos_lattice, edge_color='b', node_color='b', with_labels=node_labels)
        plt.show()
    print('NET is ready')
    return pos_lattice


def plot_accuracy(t_final: np.int_, t_for_accuracy: NDArray[np.int_], accuracy_in_t: NDArray[np.float_],
                  dataset_len: np.int_) -> None:
    """
    Plots the accuracy in time for the Iris problem

    input:
    t_final        - int, final time step
    t_for_accuracy - array of ints, times during simulation when accuracy was calculated
    accuracy_in_t  - array of floats, accuracy at simulation times "t_for_accuracy"
    dataset_len    - length of dataset used, for Iris it is 150

    output:
    plot of accuracy a.f.o time
    """
    # Add vertical lines at times where t finished cycle through dataset and targets were re-calculated
    for t in range(t_final):
        if t % dataset_len == 0:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1)

    # plot accuracy a.f.o time
    plt.plot(t_for_accuracy[1:], accuracy_in_t[1:], label='accuracy')  # plot from t=1 since starts at 0.3 and not 0

    # axes
    plt.xlabel('t', fontsize=14)  # Set x-axis label with font size
    plt.ylabel('Accuracy', fontsize=14)  # Set y-axis label with font size
    plt.title('Accuracy Over Time', fontsize=16)  # Set title with font size
    plt.ylim([0, 1])


def plot_Power(State: "Network_State") -> None:
    plt.plot(np.linspace(0, State.t, np.shape(State.Power_norm_in_t)[0]).T, State.Power_norm_in_t, color='blue')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{P}$')
    plt.yscale('log')
