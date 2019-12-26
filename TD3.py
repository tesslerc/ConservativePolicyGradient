import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from torch.distributions import Beta
# from radam import RAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

        self.l4.weight.data = fanin_init(self.l4.weight.data.size())
        self.l5.weight.data = fanin_init(self.l5.weight.data.size())
        self.l6.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, use_target_q, target_distance_weight):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_static = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_static.load_state_dict(self.actor.state_dict())
        # self.actor_optimizer = RAdam(self.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        # self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.0001, momentum=0.1)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_optimizer = RAdam(self.critic.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        # self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01, momentum=0.1)

        self.max_action = max_action
        self.use_target_q = use_target_q
        self.target_distance_weight = target_distance_weight

        self.noise_sampler = Beta(torch.FloatTensor([4.0]), torch.FloatTensor([4.0]))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_target(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor_target(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2, update_target_actor=True, update_target_q=True):

        abs_actor_loss = 0
        abs_critic_loss = 0

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d, _ = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            # noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            # noise = noise.clamp(-noise_clip, noise_clip)

            with torch.no_grad():
                target_action = self.actor_target(state)
                noise = (self.noise_sampler.rsample((action.shape[0], action.shape[1])).view(action.shape[0], action.shape[1]) * 2 - 1).to(device) * noise_clip

                target_action = (self.actor_target(next_state) + noise).clamp(-1, 1) * self.max_action

                # Compute the target Q value
                if self.use_target_q:
                    target_Q1, target_Q2 = self.critic_target(next_state, target_action)
                else:
                    target_Q1, target_Q2 = self.critic(next_state, target_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            abs_critic_loss += abs(critic_loss.item())

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                action = self.actor(state) * self.max_action
                actor_loss = -self.critic.Q1(state, action).mean() + F.mse_loss(action, target_action, reduce=True) * self.target_distance_weight

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                abs_actor_loss += abs(actor_loss.item())

                # Update the frozen target models
                if update_target_q:
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                if update_target_actor:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


        return abs_critic_loss / iterations, abs_actor_loss / iterations * policy_freq

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
