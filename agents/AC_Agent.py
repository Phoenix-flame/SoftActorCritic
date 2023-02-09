import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.autograd import Variable





class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4, device='cpu'):
        super(ActorCritic, self).__init__()
        self.device = device
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


class ACAgent:
    def __init__(self, env, logger_path='final/1'):
        self.env = env
        self.state_dim = 48
        self.action_dim = 1024
        
        self.initial_alpha = 1
        self.batch_size = 100
        self.gamma = 0.99
        self.lr = 1e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, 512)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        self.logger = Logger(logger_path)
        self.entropy_term = 0
        # self.all_rewards = []

        self.log_probs = []
        self.values = []
        self.rewards = []


    def reset(self):
        self.log_probs = []
        self.values = []
        self.rewards = []

    def step(self, state):
        value, policy_dist = self.actor_critic.forward(state)
        value = value.detach().numpy()[0,0]
        dist = policy_dist.detach().numpy() 

        action = np.random.choice(self.action_dim, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropy_term += entropy
        
        return action
       

    def train(self, obs):
        q_value, _ = self.actor_critic.forward(obs)
        q_value = q_value.detach().numpy()[0,0]

        q_values = np.zeros_like(self.values)
        for t in reversed(range(len(self.rewards))):
            q_value = self.rewards[t] + self.gamma * q_value
            q_values[t] = q_value

        #update actor critic
        values = torch.FloatTensor(self.values)
        q_values = torch.FloatTensor(q_values)
        log_probs = torch.stack(self.log_probs)
        
        advantage = q_values - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term
        print(ac_loss)
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()
        


    def save(self, idx='', path='../../'):
        torch.save(self.actor_critic.state_dict(), path + 'actor_critic_net' + str(idx) + '.h5')

    def load(self, idx='', path='./'):
        self.actor_critic.load_state_dict(torch.load(path + 'actor_critic_net' + str(idx) + '.h5', map_location='cpu'))