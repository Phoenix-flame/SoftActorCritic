import matplotlib.pyplot as plt
import numpy as np

from powergym.env_register import make_env, remove_parallel_dss
from agents.Discrete_SAC_Agent import SACAgent
import random
import os 

env = make_env("13Bus")

env.seed(0)

obs_dim = env.observation_space.shape[0]
CRB_num = ( env.cap_num, env.reg_num, env.bat_num )
CRB_dim = (2, env.reg_act_num, env.bat_act_num )
print('NumCap, NumReg, NumBat: {}'.format(CRB_num))
print('ObsDim, ActDim: {}, {}'.format(obs_dim, sum(CRB_num)))
print(env.action_space)
print('-'*80)

NUM_EPISODES = 2000

agent = SACAgent(env, logger_path=os.path.join('../../logs', '1'))

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
        # fig, _ = env.plot_graph()
        # plt.show()
        action = agent.get_next_action(obs)
        next_obs, reward, done, info = env.step(convert2multi(action))
        
        agent.train(obs, action, next_obs, reward, done)
        episode_steps += 1
        episode_reward += reward
        mask = 1 if episode_steps == env.horizon else float(not done)
        obs = next_obs

    agent.logger.log("Train/reward", episode_reward, i_episode)
    print("episode: {}, profile: {}, episode steps: {}, reward: {}".format(i_episode, load_profile_idx, episode_steps, round(episode_reward, 2)))
    
agent.save(idx=2)