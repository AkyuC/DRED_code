import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

from utils.log_utils import get_model_path, record_ppo_ratio, record_ppo_state_value


Transition = namedtuple('Transition', ['state', 'action',  'a_prob', 'reward', 'done', 'next_state'])


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        layer_init(self.fc1)
        self.fc2 = nn.Linear(512, 512)
        layer_init(self.fc2)
        self.action_head = nn.Linear(512, action_dim)
        layer_init(self.action_head)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        layer_init(self.fc1)
        self.fc2 = nn.Linear(512, 512)
        layer_init(self.fc2)
        self.state_value = nn.Linear(512, 1)
        layer_init(self.state_value)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPOCHSeletion(object):
    def __init__(self, config) -> None:
        self.config = config
        self.clip_coef = config['clip_param']
        self.max_grad_norm = config['max_grad_norm']
        self.ppo_update_time = config['ppo_update_time']
        self.buffer_capacity = config['buffer_capacity']
        self.batch_size = config['batch_size'] 

        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.entropy_coef = config['entropy_coef']
        self.vf_coef = config['vf_coef']
        self.actor_net = Actor(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic_net = Critic(state_dim=self.state_dim)
        self.buffer = [[] for _ in range(config['env_n'])]
        self.counter = 0
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=config['actor_lr'])   # 1e-5
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=config['critic_lr'])   # 1e-4

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(self.config['device'])
        if self.use_cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()
        
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state.to(self.device))
        return value.item()
    
    def store_transition(self, state, action, action_prob, reward, done, next_state, env_n):
        trans = Transition(state, action, action_prob, reward, done, next_state)
        self.buffer[env_n].append(trans)

    def choose_abstract_action(self, current_state, action_mask=False):
        FloatTensor = torch.FloatTensor
        state = torch.from_numpy(current_state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state.type(FloatTensor).to(self.device))

        if action_mask is not False:
            # mask operation
            action_prob_np = action_prob.data.cpu().numpy()
            action_prob_np[:, action_mask] = 0
            action_prob = torch.FloatTensor(action_prob_np)

        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item(), c.entropy()

    def update(self):
        if len(self.buffer[0]) < self.batch_size: return False, False
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

        closs_list = []
        aloss_list = []
        state = (torch.tensor([[t.state.tolist() for t in b] for b in self.buffer], dtype=torch.float)).type(FloatTensor)
        actions = (torch.tensor([[t.action for t in b] for b in self.buffer], dtype=torch.long)).type(LongTensor)
        rewards = torch.tensor([[t.reward for t in b] for b in self.buffer]).type(FloatTensor)
        dones = (torch.tensor([[t.done + 0 for t in b] for b in self.buffer], dtype=torch.float)).type(FloatTensor)
        next_state = (torch.tensor([[t.next_state.tolist() for t in b] for b in self.buffer], dtype=torch.float)).type(FloatTensor)
        old_action_prob = (torch.tensor([[t.a_prob for t in b] for b in self.buffer], dtype=torch.float)).type(FloatTensor)

        next_values = torch.squeeze(self.critic_net(next_state))
        values = torch.squeeze(self.critic_net(state))
        advantages = torch.zeros_like(rewards).type(FloatTensor)
        lastgaelam = 0
        for t in reversed(range(self.config['env_step'])):
            terminated_flag = (1.0 - dones[:,t])
            delta = rewards[:,t] + self.gamma * next_values[:,t] * terminated_flag - values[:,t]
            advantages[:,t] = lastgaelam = delta + self.gamma * self.gae_lambda * terminated_flag * lastgaelam
        returns = advantages + values

        b_obs = state.reshape((-1, self.state_dim)).type(FloatTensor).detach()
        b_old_probs = old_action_prob.reshape(-1, 1).type(FloatTensor).detach()
        b_actions = actions.reshape(-1, 1).type(LongTensor).detach()
        b_advantages = advantages.reshape(-1, 1).type(FloatTensor).detach()
        b_returns = returns.reshape(-1, 1).type(FloatTensor).detach()

        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(b_obs))), self.batch_size, False):
                new_action_prob = self.actor_net(b_obs[index])
                entropy = Categorical(new_action_prob).entropy()
                new_action_prob = new_action_prob.gather(1, b_actions[index])
                ratio = (new_action_prob/b_old_probs[index])
                record_ppo_ratio(float(torch.mean(ratio)), self.config)

                mb_advantages = b_advantages[index]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                entropy_loss = entropy.mean()
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.entropy_coef * entropy_loss
                aloss_list.append(float(pg_loss))
                self.actor_optimizer.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Value loss
                new_value = (self.critic_net(b_obs[index]))
                record_ppo_state_value(float(torch.mean(new_value)), self.config)
                v_loss = ((new_value - b_returns[index]) ** 2).mean()
                closs_list.append(float(v_loss))
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]  # clear experience
        self.buffer = [[] for _ in range(self.config['env_n'])]
        return aloss_list, closs_list

    def save(self, cnt_episode):
        path = get_model_path(self.config, True)
        torch.save(self.actor_net.state_dict(), path + f'/actor{cnt_episode}.ckpt')
        torch.save(self.critic_net.state_dict(), path + f'/critic{cnt_episode}.ckpt')

    def save_best(self, cnt_episode):
        path = get_model_path(self.config, True)
        if os.path.exists(path + '/best_episode'):
            os.system(f"rm {path}/best_episode")
        with open(path + '/best_episode', 'w+') as f:
            f.write(str(cnt_episode))
        torch.save(self.actor_net.state_dict(), path + f'/actor_best.ckpt')
        torch.save(self.critic_net.state_dict(), path + f'/critic_best.ckpt')

    def load(self, cnt_episode):
        path = get_model_path(self.config, True)
        self.actor_net.load_state_dict(torch.load(path + f'/actor{cnt_episode}.ckpt'))
        self.critic_net.load_state_dict(torch.load(path + f'/critic{cnt_episode}.ckpt'))

    def learn(self):
        if len(self.buffer) >= self.batch_size:
            aloss, closs = self.update()
            return aloss, closs
        return False, False