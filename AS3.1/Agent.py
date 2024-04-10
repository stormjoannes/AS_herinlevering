import numpy as np



class Agent:

    def __init__(self, policy: classmethod, memory: int, discount: float):
        """
        Set class values

            Parameters:
                 policy(classmethod): Current policy class
                 memory(int): Max amount of memory that is stored
                 discount(int): Amount of discount for the agent, impacts the importance of values/rewards
        """
        self.policy = policy
        self.memory = memory
        self.discount = discount

    def train(self):
        """
        Predict actions, calculate new values and train the model.
        """
        # Sample random minibatch of transitions from D
        states, actions, rewards, next_states, terminated = self.memory.sample()
        actions_states = self.policy.model.predict(states)

        next_state_q_value = self.policy.model.predict(next_states)
        q_values = np.copy(actions_states)

        # ----------------------- OLD -----------------------
        # Loop for each row and calculate action state value
        for row, action in zip(range(len(actions_states[0])), actions):
            # set y_j = r_j for terminal φ_j+1, otherwise y_j = r_j + γ max_a' Q(φ_j+1, a'; θ)
            if terminated[row]:
                q_values[0][row][action] = rewards[row]
            else:
                q_values[0][row][action] = rewards[row] + self.discount * np.max(next_state_q_value[0][row])
        # ----------------------- OLD -----------------------

        # ----------------------- NEW -----------------------
        # for i in range(len(actions)):
        #     action = actions[i]
        #     reward = rewards[i]
        #     terminal = terminated[i]
        #
        #     # set y_j = r_j for terminal φ_j+1
        #     if terminal:
        #         q_values[0][i][action] = reward
        #     else:
        #         q_values[0][i][action] = reward + self.discount * np.max(next_state_q_value[0][i])
        # ----------------------- NEW -----------------------

        self.policy.model.train_on_batch(states, q_values)
        # Perform a gradient descent step on (y_j - Q(φ_j, a_j; θ))^2
        self.policy.decay()
