from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import Callable, Union, Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle

import functions


# ===================================================
# Class - User Variables
# ===================================================


class User_Variables:
    """
    Class with variables given by user.
    These remain the same along the simulation
    """
    def __init__(self, iterations: int, Nin: int, extraNin: int, Ninter: int, Nout: int, extraNout: int,
                 gamma: NDArray[np.float_], R_update: str, use_p_tag: bool, include_Power: bool, lam: np.float_,
                 supress_prints: bool, bc_noise: float, access_interNodes: bool, task_type: str,
                 measure_accuracy_every: Optional[int] = None) -> None:

        self.iterations: int = iterations
        self.Nin: int = Nin
        self.extraNin: int = extraNin
        self.Nout: int = Nout
        self.extraNout: int = extraNout
        self.Ninter: int = Ninter
        self.NN: int = Nin + extraNin + Nout + extraNout + Ninter
        self.gamma: NDArray[np.float_] = gamma
        self.use_p_tag: bool = use_p_tag
        self.include_Power = include_Power
        self.lam = lam
        if use_p_tag:
            self.loss_fn: Union[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                          Optional[np.float_], Optional[np.float_], Optional[np.float_]], np.ndarray],
                                Callable[[np.ndarray, np.ndarray, Optional[np.float_], Optional[np.float_]],
                                         np.ndarray]] = functions.loss_fn_2samples
        else:
            self.loss_fn = functions.loss_fn_1sample
        self.R_update: str = R_update  # 'R_propto_dp' if R=gamma*delta_p
                                       # 'deltaR_propto_dp' if deltaR=gamma*delta_p, gamma should be small
                                       # 'R_propto_Q' if deltaR=gamma*Q where Q is flow velocity
                                       # 'deltaR_propto_Q' if R=gamma*Q where Q is flow velocity
                                       # 'deltaR_propto_Power' if R=gamma*P where P is power dissipation
        self.supress_prints: bool = supress_prints
        self.bc_noise: float = bc_noise
        self.access_interNodes: bool = access_interNodes
        self.task_type: str = task_type
        if task_type == 'Iris_classification' and self.Nin != 4 and self.Nout != 3:
            print('mismatched # of inputs and outputs for Iris classification. correcting accordingly to Nin=4 Nout=3')
            self.Nin = 4
            self.Nout = 3
        if measure_accuracy_every is not None:
            self.measure_accuracy_every = measure_accuracy_every

    def create_dataset_and_targets(self, random_state, M_values: Optional[NDArray[np.float_]] = None,
                                   train_size: Optional[int] = None) -> None:
        """
        creates the matrix which defines the task, i.e. p_out=M*p_in

        inputs:
        random_state - int, seed for random function
        M_values     - 1D np.ndarray of all values to be inserted to M, consecutively, regardless of structure
        train_size   - how many samples for train dataset

        outputs:
        M: np.ndarray sized [Nout, Nin], matrix defining the task p_out=M*p_in
        """
        if self.task_type == 'Regression':
            if M_values is None:
                print('M not specified, assigning zeros')
                M_values = zeros([self.Nin*self.Nout], dtype=np.float_)
            elif np.size(M_values) != self.Nin*self.Nout:
                print('input M mismatches output and input')
            np.random.seed(42)  # Set seed
            # Generate random numbers as dataset and multiply by task matrix M
            self.dataset: NDArray[np.float_] = np.random.uniform(low=0.0, high=2.0, size=[self.iterations, self.Nin])
            self.M: np.ndarray = M_values[0:self.Nout*self.Nin].reshape(self.Nout, self.Nin)
            self.targets: NDArray[np.float_] = np.matmul(self.dataset, self.M.T)
            self.X_train = copy.copy(self.dataset)  # train and test are the full dataset, no difference
            self.y_train = copy.copy(self.targets)  # train and test are the full dataset, no difference
            self.X_test = copy.copy(self.dataset)  # train and test are the full dataset, no difference
            self.y_test = copy.copy(self.targets)  # train and test are the full dataset, no difference
        elif self.task_type == 'Iris_classification':
            # Load the Iris dataset
            iris = load_iris()

            # Min-Max Scale dataset to [0, 5]
            min_max_scaler = MinMaxScaler(feature_range=(0, 5))
            self.dataset = min_max_scaler.fit_transform(iris['data'])

            # One-hot encode the labels
            encoder = OneHotEncoder(sparse_output=False, categories='auto')
            targets_reshaped = iris['target'].reshape(-1, 1)  # Reshape for the encoder
            self.targets = encoder.fit_transform(targets_reshaped)

            # Split the dataset: 30 for training, the rest for testing, Ensure balanced class representation
            if train_size:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.targets,
                                                                                        train_size=train_size,
                                                                                        random_state=random_state,
                                                                                        stratify=iris['target'])
            else:  # or use all dataset as train and test if not specified
                self.X_train = shuffle(copy.copy(self.dataset), random_state=random_state)
                self.X_test = shuffle(copy.copy(self.dataset), random_state=random_state)
                self.y_train = shuffle(copy.copy(self.targets), random_state=random_state)
                self.y_test = shuffle(copy.copy(self.targets), random_state=random_state)

            # Calculate the means for each class using only the training set
            # First, decode the one-hot encoded y_train back to numerical labels
            y_train_dec = np.argmax(self.y_train, axis=1)

            # Calculate means for each class based on training set
            means = [np.mean(self.X_train[y_train_dec == i], axis=0) for i in range(3)]
            self.means = np.array(means)

    def create_noise_for_extras(self) -> None:
        """
        add desc
        """
        dataset_size = np.shape(self.X_train)[0]

        def generate_uniform_noise(size: list[int]) -> NDArray[np.float_]:
            """
            add descr
            """
            return (np.random.uniform(low=0.0, high=1.0, size=size) - 0.5) * self.bc_noise

        if self.extraNin != 0:  # generate noise to add to extra input nodes
            self.noise_in = generate_uniform_noise([dataset_size, self.extraNin])
        else:
            # print('no extra input nodes, no noise added')
            self.noise_in = zeros([dataset_size, self.extraNin])

        if self.Ninter != 0:  # generate noise to add to extra input nodes
            self.noise_inter = generate_uniform_noise([dataset_size, self.Ninter])
        else:
            # print('no inter nodes, no noise added')
            self.noise_inter = zeros([dataset_size, self.extraNin])

        if self.extraNout != 0:  # generate noise to add to extra output nodes
            self.noise_out = generate_uniform_noise([dataset_size, self.extraNout])
        else:
            # print('no extra output nodes, no noise added')
            self.noise_out = zeros([dataset_size, self.extraNout])

    def assign_alpha_vec(self, alpha: float) -> None:
        """
        assign the alpha vector, in the form of array of [Nout], to the User_Variables

        inputs:
        alpha: float of the learning rate alpha
        """
        if isinstance(alpha, float):
            self.alpha_vec: NDArray[np.float_] = np.tile(alpha, (self.Nout,))
        else:
            print('wrong type for alpha, should be float')
