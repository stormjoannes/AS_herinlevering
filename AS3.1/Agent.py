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
        # Sample random minibatch of transitions et = (st, at, rt, st+1) from D
        states, actions, rewards, next_states, terminated = self.memory.sample()

        # Predict Q values for current and next states
        current_q_values = self.policy.model.predict(states)
        next_q_values = self.policy.model.predict(next_states).reshape(self.memory.batch_size, -1)

        # Update Q values
        # Q* (st, at) = rt + γ Q0'(st+1, argmax_a Q(st+1, a; θ); θ)
        target_q_values = np.copy(current_q_values).reshape(self.memory.batch_size, -1)
        batch_indices = np.arange(self.memory.batch_size)
        target_q_values[batch_indices, actions] = rewards + (1 - terminated) * self.discount * np.max(next_q_values,
                                                                                                      axis=1)

        # Flatten states to match target_q_values shape
        states = states.reshape(self.memory.batch_size, -1)

        # Train the model
        # perform a gradient descent step on (y_j - Q(φ_j, a_j; θ))^2
        self.policy.model.train_on_batch(states, target_q_values)
        self.policy.decay()
