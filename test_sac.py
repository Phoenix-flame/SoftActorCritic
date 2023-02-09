import matplotlib.pyplot as plt
import numpy as np

from powergym.env_register import make_env, remove_parallel_dss
from agents.SAC_Agent import SACAgent
import random
import os 

env = make_env("13Bus")
env.seed(0)

NUM_EPISODES = 1

agent = SACAgent(env, logger_path=os.path.join('../../logs', '1'))
agent.load(idx='1', path='../../')

def convert2multi(action):
    return np.unravel_index(action, (2, 2, 4, 4, 4, 4))

profiles = list(range(env.num_profiles))
for i_episode in range(NUM_EPISODES):
    episode_reward = 0
    episode_steps = 0
    done = False
    load_profile_idx = random.choice(profiles)
    obs = env.reset(load_profile_idx = load_profile_idx)
    while not done:
        fig, _ = env.plot_graph()
        plt.savefig(f"../../{episode_steps}.png")
        action = agent.get_next_action(obs, evaluation_episode=True)
        next_obs, reward, done, info = env.step(convert2multi(action))
        
        episode_steps += 1
        episode_reward += reward
        obs = next_obs

    agent.logger.log("Train/reward", episode_reward, i_episode)
    print("episode: {}, profile: {}, reward: {}".format(i_episode, load_profile_idx, round(episode_reward, 2)))
