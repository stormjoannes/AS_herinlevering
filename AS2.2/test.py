import numpy as np
from Maze import Maze


class SARSA:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.maze = maze
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_values = np.zeros((len(maze.grid), len(maze.actions)))

    def choose_action(self, position):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.maze.actions))  # Exploration: choose random action
        else:
            return np.argmax(self.q_values[position])  # Exploitation: choose action with max Q-value

    def update_q_value(self, position, action, reward, next_position, next_action):
        current_q_value = self.q_values[position, action]
        next_q_value = self.q_values[next_position, next_action]
        td_target = reward + self.discount_factor * next_q_value
        td_error = td_target - current_q_value
        self.q_values[position, action] += self.learning_rate * td_error

# Create maze instance
maze = Maze()
maze.create_maze_values()

# Initialize SARSA agent
agent = SARSA(maze)

# Simulate SARSA learning
num_episodes = 1000
for episode in range(num_episodes):
    position = (0, 0)  # Initial position
    action = agent.choose_action(position)
    done = False
    while not done:
        next_position = maze.stepper(position, action)
        reward = maze.rewards[next_position]
        next_action = agent.choose_action(next_position)
        agent.update_q_value(maze.grid_index[position], action, reward, maze.grid_index[next_position], next_action)

        position = next_position
        action = next_action

        if position in maze.terminal_states:
            done = True

# Print learned Q-values
print("Learned Q-values:")
print(agent.q_values)