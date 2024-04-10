import random
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import l2

import numpy as np


class Policy:
    def __init__(self, epsilon: float):
        """
        Defines the values of class Policy.

            Parameters:
                 epsilon(float): Current epsilon
        """
        self.epsilon = epsilon
        self.model = None

    def select_action(self, state: np.ndarray) -> int:
        """
        Implementing a partial random agent.
        This way it sometimes uses the model, but also discovers new paths by the randomness
        How longer the model runs (more trained) the less the chance is to choose a random action because of
        the decaying epsilon.

            Parameter:
                state(np.ndarray):

            Return:
                action(int): action to take
        """
        random_epsilon = round(random.random(), 2)
        if random_epsilon < self.epsilon:
            # print('select aciton 1')
            action = random.choice((0, 1, 2, 3))
            return action

        else:
            # print('select aciton 2')
            state = np.array([state])
            # print(state, 'state', state.shape)
            output = self.model.predict(state)
            action = np.argmax(output)
            return action

    def decay(self):
        """
        Decaying epsilon
        """
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def setup_model(self, dimensions: int, actions: list, learning_rate: float, regularization_factor: float = 0.001):
        """
        Define model settings

            Parameters:
                 dimensions(int): Amount of dimensions used for the dense layer
                 actions(list): The possible actions to do
                 learning_rate(float): Learning rate for the model
                 regularization_factor(float): Regularization factor for the model
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(dimensions, activation="relu", input_shape=(None, 8), kernel_regularizer=l2(regularization_factor)))
        model.add(layers.Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)))
        model.add(layers.Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)))
        model.add(layers.Dense(actions))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())

        self.model = model

