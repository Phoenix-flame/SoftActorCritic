import numpy as np
import torch

from .utilities.Network import Network
from .utilities.ReplayBuffer import ReplayBuffer
from .utilities.logger import Logger

class SACAgent:
    def __init__(self, env, logger_path='final/1'):
        self.env = env
        self.state_dim = 48
        self.action_dim = 1024
        
        self.initial_alpha = 1
        self.batch_size = 100
        self.gamma = 0.99
        self.lr = 1e-4
        self.soft_update_factor = 0.01

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic_net = Network(n_inputs=self.state_dim, n_outputs=self.action_dim).to(self.device)
        self.critic_net2 = Network(n_inputs=self.state_dim, n_outputs=self.action_dim).to(self.device)
        self.critic_optimiser = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lr)

        self.critic_target = Network(n_inputs=self.state_dim, n_outputs=self.action_dim).to(self.device)
        self.critic_target2 = Network(n_inputs=self.state_dim, n_outputs=self.action_dim).to(self.device)

        self.soft_update_target_networks(tau=1.)

        self.actor = Network(n_inputs=self.state_dim, n_outputs=self.action_dim, output_activation=torch.nn.Softmax(dim=1)).to(self.device)
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.env)

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        self.log_alpha = torch.tensor(np.log(self.initial_alpha), requires_grad=True, device=self.device)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.logger = Logger(logger_path)

    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            action = self.get_action_deterministically(state)
        else:
            action = self.get_action_nondeterministically(state)
        return action

    def get_action_nondeterministically(self, state):
        action_prob = self.get_action_probabilities(state)
        action = np.random.choice(range(self.action_dim), p=action_prob)
        return action

    def get_action_deterministically(self, state):
        action_prob = self.get_action_probabilities(state)
        action = np.argmax(action_prob)
        return action


    def train(self, state, action, next_state, reward, done):
        transition = (state, action, reward, next_state, done)

        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        self.replay_buffer.add_transition(transition)

        if self.replay_buffer.get_size() >= self.batch_size:
            batch = self.replay_buffer.sample_minibatch(self.batch_size)
            batch = list(map(list, zip(*batch)))

            states_tensor = torch.tensor(np.array(batch[0]), device=self.device)
            actions_tensor = torch.tensor(np.array(batch[1]), device=self.device, dtype=torch.int64)
            rewards_tensor = torch.tensor(np.array(batch[2]), device=self.device).float()
            next_states_tensor = torch.tensor(np.array(batch[3]), device=self.device)
            done_tensor = torch.tensor(np.array(batch[4]), device=self.device)

            critic_loss, critic2_loss = self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_prob = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_prob)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks(tau=self.soft_update_factor)
        

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_prob, log_action_prob = self.get_action_info(next_states_tensor)
            next_q_target = self.critic_target.forward(next_states_tensor)
            next_q_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_prob * (
                    torch.min(next_q_target, next_q_target2) - self.alpha * log_action_prob
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.gamma*soft_state_values

        soft_q_values = self.critic_net(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_net2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        critic_se = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_se = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_se, critic2_se)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_se.mean()
        critic2_loss = critic2_se.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_prob, log_action_prob = self.get_action_info(states_tensor)
        q_values_local = self.critic_net(states_tensor)
        q_values_local2 = self.critic_net2(states_tensor)
        inside_term = self.alpha * log_action_prob - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_prob * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_prob

    def temperature_loss(self, log_action_prob):
        alpha_loss = -(self.log_alpha * (log_action_prob + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_prob = self.actor.forward(states_tensor)
        z = action_prob == 0.0
        z = z.float() * 1e-8
        log_action_prob = torch.log(action_prob + z)
        return action_prob, log_action_prob

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        action_prob = self.actor.forward(state_tensor)
        return action_prob.squeeze(0).cpu().detach().numpy()

    def soft_update_target_networks(self, tau):
        self.soft_update(self.critic_target, self.critic_net, tau)
        self.soft_update(self.critic_target2, self.critic_net2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_net(state)
        q_values2 = self.critic_net2(state)
        return torch.min(q_values, q_values2)

    def save(self, idx='', path='../../'):
        print(path + 'actor_net' + str(idx) + '.h5')
        torch.save(self.actor.state_dict(), path + 'actor_net' + str(idx) + '.h5')
        torch.save(self.critic_net.state_dict(), path + 'critic_net' + str(idx) + '.h5')

    def load(self, idx='', path='./'):
        self.actor.load_state_dict(torch.load(path + 'actor_net' + str(idx) + '.h5', map_location='cpu'))
        self.critic_net.load_state_dict(torch.load(path + 'critic_net' + str(idx) + '.h5', map_location='cpu'))