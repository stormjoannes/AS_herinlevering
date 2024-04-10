"""In this file we make sure everything is connected and run properly"""

from Maze import Maze
from Agent import Agent
from Policy import Policy

delta_threshold = 0.01
start_position = (3, 2)

maze = Maze()
maze.position = start_position
maze.create_maze_values()

policy = Policy(maze)

agent = Agent(start_position, maze, policy, delta_threshold)
agent.value_iteration()

# Test act function
amount_acts = 0
while 1 < float('inf'):
    amount_acts += 1
    print("action:", amount_acts)
    if agent.act() == 'terminal':
        break
