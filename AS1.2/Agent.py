"""In this file we will iterate over values"""

import matplotlib.pyplot as plt
import math


class Agent:

    def __init__(self, position: tuple, Maze: classmethod, Policy: classmethod, delta_threshold: float):
        """Set all class values"""
        self.position = position
        self.maze = Maze
        self.policy = Policy
        self.delta_threshold = delta_threshold

    def value_iteration(self):
        """
        Iterating over the maze grid and calculate each value for each iteration.
        """
        # Make sure the first time iteration it will always go into the while loop.
        delta = self.delta_threshold + 1
        iteration = 0
        while delta >= self.delta_threshold:
            print(f"Begin delta: {delta}")
            # Set delta low so that first loop delta will always be overwritten.
            delta = 0
            for state in self.maze.grid:
                # Check if state isn't a terminal state.
                if state not in self.maze.terminal_states:
                    # Gets the coords of the states surrounding the current state.
                    new_value = self.policy.select_action(state, iteration)
                    self.maze.grid[state].append(new_value)

                    # Get difference from old to new value
                    new_delta = abs(self.maze.grid[state][iteration] - new_value)
                    delta = new_delta if new_delta > delta else delta
                else:
                    # If state is terminal state, new value is always zero.
                    self.maze.grid[state].append(0)

            print(f"End delta: {delta}")
            self.print_iteration(iteration)
            iteration += 1

        self.plot_values(3, 3, "1.2")

    def act(self):
        """
        Move agent through the maze
        """
        print("Position: ", self.position)
        surr_states = self.maze.surrounding_states(self.position)
        action = self.policy.choose_action(self.position, surr_states)

        # als de waarde van de beste surrounding state niet beter is dan huidige waarde, terminal
        if not action:
            print("Terminal state")
            return 'terminal'
        else:
            new_pos = self.maze.stepper(self.position, action[0])
            self.position = new_pos
            print("New position", self.position, "\n")
            return 'not terminal'

    def print_iteration(self, iteration):
        """
        Quick hardcoded print to show values for each iteration.
        """
        values = []
        for coord in self.maze.grid:
            values.append(self.maze.grid[coord][iteration])

        values = [round(value, 1) for value in values]

        print("Iteration: ", iteration)
        print(values[0: 4])
        print(values[4: 8])
        print(values[8: 12])
        print(values[12: 16], "\n")

    def plot_values(self, tot_fig_rows, tot_fig_columns, plt_name):
        """
        Plot the values in a state transition matrix
        """
        iterations = len(self.maze.grid[0, 0])
        rows = math.ceil(math.sqrt(len(self.maze.grid)))
        cols = math.ceil(math.sqrt(len(self.maze.grid)))

        fig, axs = plt.subplots(tot_fig_rows, tot_fig_columns, figsize=(10, 9))
        fig.suptitle('Heatmaps for Each Iteration')

        for i in range(iterations):
            values = []
            for r in range(rows):
                row_values = []
                for c in range(cols):
                    key = (r, c)
                    row_values.append(self.maze.grid[key][i])
                values.append(row_values)

            ax = axs[i // tot_fig_rows, i % tot_fig_columns]
            ax.imshow(values, cmap='viridis', interpolation='nearest')

            for r in range(rows):
                for c in range(cols):
                    color = 'black' if values[r][c] > 30 else 'white'
                    ax.text(c, r, values[r][c], ha='center', va='center', color=color)

            ax.set_title(f'Iteration {i + 1}')
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.set_xticklabels([str(j) for j in range(cols)])
            ax.set_yticklabels([str(rows - j - 1) for j in range(rows)][::-1])  # Reversed y-axis tick labels

        plt.tight_layout()
        plt.savefig(f'../images/AS{plt_name}_visualization.png')
        plt.show()
