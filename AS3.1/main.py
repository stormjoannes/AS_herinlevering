import gym
from Agent import Agent
from Policy import Policy
from Memory import Memory, Transition
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

episodes = 500
max_steps_episode = 1500
max_memory_size = 32000
batch_size = 64
learning_rate = 0.0005
update_rate = 4
scores = []
average_scores = []

# Initialize replay memory D to capacity N
memory = Memory(batch_size, max_memory_size)
policy = Policy(1)
agent = Agent(policy, memory, 0.99)
actions = env.action_space.n
dimensions = env.observation_space.shape
print(dimensions)
agent.policy.setup_model(dimensions[0], actions, learning_rate)
print("setup model")

for episode in range(episodes):
    observation, info = env.reset()
    steps = 0
    score = 0
    terminated = False
    while steps < max_steps_episode and not terminated:
        # With probability epsilon select a random action a_t, otherwise select a_t = argmax_a Q(φ_t, a; θ)
        action = agent.policy.select_action(observation)
        # Execute action a_t in emulator and observe reward r_t and image x_t+1
        next_observation, reward, terminated, truncated, info = env.step(action)
        # Store transition φ_t, a_t, r_t, φ_t+1 in D
        agent.memory.store(Transition(observation, action, reward, next_observation, terminated))
        steps += 1
        score += reward
        observation = next_observation
        # Gather 4 iterations data so batch size of 32 can always be filled
        if len(agent.memory.deque) >= batch_size and (steps + 1) % update_rate == 0:
            agent.train()

    print('\n', "Scores: ", scores)
    print("AVG_Scores: ", average_scores, '\n')
    scores.append(score)
    # Gemiddelde van laatste 50 episodes for smoothness
    last_scores = np.mean(scores[-50:])
    average_scores.append(last_scores)
    print("episode", episode, f"score {score}", f"average score {last_scores}")

env.close()

plt.plot(np.arange(episodes), np.array(average_scores), label="Average score")
plt.xticks(np.arange(0, episodes+1, 100))
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.savefig(f'AS_3.1_visualization.png')
plt.show()

# pseudocode for deap q learning lundar lander
# initialize replay memory D to capacity N
# initialize action-value function Q with random weights
# for episode = 1, M do
#     initialize sequence s_1 = {x_1} and preprocessed sequenced φ_1 = φ(s_1)
#     for t = 1, T do
#         with probability ε select a random action a_t
#         otherwise select a_t = argmax_a Q(φ(s_t), a; θ)
#         execute action a_t in emulator and observe reward r_t and image x_t+1
#         set s_t+1 = s_t, a_t, x_t+1 and preprocess φ_t+1 = φ(s_t+1)
#         store transition (φ_t, a_t, r_t, φ_t+1) in D
#         sample random minibatch of transitions (φ_j, a_j, r_j, φ_j+1) from D
#         set y_j = r_j for terminal φ_j+1
#         otherwise y_j = r_j + γ max_a' Q(φ_j+1, a'; θ)
#         perform a gradient descent step on (y_j - Q(φ_j, a_j; θ))^2
#     end for
# end for

