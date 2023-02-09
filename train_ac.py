import matplotlib.pyplot as plt
import numpy as np

from powergym.env_register import make_env, remove_parallel_dss
import random
import os 

import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from agents.utilities.logger import Logger

env = make_env("13Bus")

env.seed(0)

obs_dim = env.observation_space.shape[0]
CRB_num = ( env.cap_num, env.reg_num, env.bat_num )
CRB_dim = (2, env.reg_act_num, env.bat_act_num )
print('NumCap, NumReg, NumBat: {}'.format(CRB_num))
print('ObsDim, ActDim: {}, {}'.format(obs_dim, sum(CRB_num)))
print(env.action_space)
print('-'*80)

def convert2multi(action):
    return np.unravel_index(action, (2, 2, 4, 4, 4, 4))



# hyperparameters
hidden_size = 512
learning_rate = 3e-4


# Constants
gamma = 0.99
max_episodes = 2000

logger = Logger("../../logs/ac_1")

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist



num_inputs = env.observation_space.shape[0]
num_outputs = 1024

actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)


entropy_term = 0
profiles = list(range(env.num_profiles))
for i_episode in range(max_episodes):
    log_probs = []
    values = []
    rewards = []
    done = False
    load_profile_idx = random.choice(profiles)
    obs = env.reset(load_profile_idx = load_profile_idx)
    while not done:
        value, policy_dist = actor_critic.forward(obs)
        value = value.detach().numpy()[0,0]
        dist = policy_dist.detach().numpy() 

        action = np.random.choice(num_outputs, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))
        next_obs, reward, done, _ = env.step(convert2multi(action))

        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        entropy_term += entropy
        obs = next_obs

    q_value, _ = actor_critic.forward(next_obs)
    q_value = q_value.detach().numpy()[0,0]
    logger.log("Train/Reward", np.mean(rewards), i_episode)
    if i_episode % 10 == 0:                    
        print("episode: {}, reward: {} \n".format(i_episode, np.sum(rewards)))
    

    q_values = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        q_value = rewards[t] + gamma * q_value
        q_values[t] = q_value

    values = torch.FloatTensor(values)
    q_values = torch.FloatTensor(q_values)
    log_probs = torch.stack(log_probs)
    
    advantage = q_values - values
    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

    ac_optimizer.zero_grad()
    ac_loss.backward()
    ac_optimizer.step()

        
    