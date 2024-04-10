"""In this file we define our policy class"""
import random


class Policy:

    def __init__(self, maze):
        """
        Defines the values of class Policy.

            Parameters:
                maze(class): current maze class
        """
        self.maze = maze
        self.discount = 1

    def select_action(self, position: tuple, iteration: int) -> float:
        """
        Select max value state, from states touching the given state.

            Parameters:
                position(tuple): Combination of x and y-axis position
                iteration(int): Current iteration

            Return:
                new_values(float): New value of the position
        """

        opt_1 = self.maze.stepper(position, 0)  # up
        opt_2 = self.maze.stepper(position, 1)  # down
        opt_3 = self.maze.stepper(position, 2)  # right
        opt_4 = self.maze.stepper(position, 3)  # left
        options = [opt_1, opt_2, opt_3, opt_4]

        new_value = self.value_func(options, self.discount, iteration)

        return new_value

    def value_func(self, next_states: list, discount: float, iteration: int) -> float:
        """
        Calculate the new values for the given next states.

            Parameters:
                next_states(list):
                discount(float): Current discount
                iteration(int): Current iteration

            Return:
                next_value(float): greedy next value
        """
        next_values = []
        for state in next_states:
            val = self.maze.rewards[state] + discount * self.maze.grid[state][iteration]
            next_values.append(val)

        return max(next_values)

    def choose_action(self, position: tuple, surr_states: list) -> int:
        """
        Check which action is needed to get to the best surrounding state.

            Parameters:
                position(tuple): Current position
                surr_states(list): Values from surrounding states

            Return:
                action(int): Action to take
        """
        highest_state = max(surr_states, key=surr_states.get)

        pos_action = [highest_state[0] - position[0], highest_state[1] - position[1]]
        action = [key for key, action_position in self.maze.actions.items() if action_position == pos_action]
        return action

    def decide_action_value(self, epsilon: float, surr_values: list) -> int:
        """
        Decide action, depending on the epsilon it can differ if it is random or not.
        In the case of a decaying epsilon, each next episode will have a lower chance of rng.

            Parameters:
                epsilon(float): Current epsilon
                surr_values(list): Surrounding values of state

            Return:
                action(int): Chosen action to take
        """
        rng = round(random.random(), 2)
        # If random number is lower than epsilon, pick a random action
        if rng < epsilon:
            choice = random.choice([0, 1, 2, 3])
            return choice

        else:
            # Choose the action to make based on the highest surrounding value
            action_value = max(surr_values)
            action = surr_values.index(action_value)

            return action
