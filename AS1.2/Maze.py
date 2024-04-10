"""In this file we define create our maze"""


class Maze:

    def __init__(self):
        """Set all class values"""
        self.rewards = {}
        self.grid = {}
        self.terminal_states = []
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}

    def stepper(self, position: tuple, action: int) -> tuple:
        """
        Get the new position based on the action

            Parameters:
                position(tuple): Current x and y-axis position
                action(int): The chosen action to take

            Return:
                tuple: Next x and y-axis position
        """
        movement = self.actions[action]
        new_position = (position[0] + movement[0], position[1] + movement[1])

        if new_position in self.grid:
            return new_position
        else:
            return position

    def surrounding_states(self, position: tuple) -> dict:
        """
        Get the surrounding states for current position

            Parameters:
                position(tuple): Current x and y-axis position

            Return:
                states(dict): list with for each state the surrounding rewards and values
        """
        states = {}
        for action in self.actions.keys():
            state = self.stepper(position, action)
            states[state] = [self.rewards[state], self.grid[state][-1]]
        return states

    def fillDict(self, value: list, sizeHorizontal: int = 4, sizeVertical: int = 4) -> dict:
        """
        Filling a dictionary with coördinates.

            Parameters:
                value(list): List of begin value for each position
                sizeHorizontal(int): Horizontal size of the maze
                sizeVertical(int): Vertical size of the maze

            Return:
                dict(dict): Dictionary with each position defined
        """
        index = 0
        dict = {}
        for vertical in range(sizeVertical):
            for horizontal in range(sizeHorizontal):
                coord = (vertical, horizontal)
                dict[coord] = value[index]
                index += 1

        return dict

    def create_maze_values(self):
        """
        Create the values and rewards in a dictionary.
        """
        values = [[0] for _ in range(16)]

        rewards = [-1, -1, -1, 40,
                   -1, -1, -10, -10,
                   -1, -1, -1, -1,
                   10, -2, -1, -1]

        self.terminal_states = [(0, 3), (3, 0)]

        self.grid, self.rewards = self.fillDict(values), self.fillDict(rewards)
