from envs.power_system_2 import Power_system_2
from nets.policy_net import PolicyNetwork
from nets.vertex_policy_net import VertexPolicyNetwork
from nets.value_net import ValueNetwork
from utils.replay_buffer import ReplayBuffer
from algos.ddpy import DDPG

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pickle
import os

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

env = Power_system_2()
obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 256
num_episodes = 200
num_steps = 100  # 5 seconds
batch_size = 128

seed = 10

parent_dir = os.getcwd()


def train_agent(path,
                env,
                agent,
                seed=0,
                num_episodes=50,
                num_steps=100,
                batch_size=128,
                replay_buffer_size=1000000):

    if not os.path.isdir(path):
        os.makedirs(path)
    os.chdir(path)

    env.seed(seed)
    random.seed(seed)

    pickle.dump(agent.policy_net, open('first_policy.pickle', 'wb'))

    replay_buffer = ReplayBuffer(replay_buffer_size)

    rewards = []
    max_angle = []
    ave_angle = []
    max_penalties = []
    ave_penalties = []
    all_traj = []
    ave_w = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        max_th = 0
        ave_th = 0
        max_penalty = 0
        ave_penalty = 0
        traj = []
        W = 0
        for step in range(num_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action,5,episode,step,seed)
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > batch_size:
                agent.train_step(replay_buffer=replay_buffer, batch_size=batch_size)

            state = next_state
            episode_reward += reward
            th = np.arccos(state[0]) * np.sign(state[1])
            max_th = max(max_th, abs(th))
            ave_th += abs(th)
            traj.append(th)

        rewards.append(episode_reward)
        max_penalties.append(max_penalty)
        max_angle.append(max_th)
        ave_angle.append(ave_th / num_steps)
        ave_penalties.append(ave_penalty / num_steps)
        all_traj.append(traj)
        ave_w.append(W/num_steps)
        print('Episode: ' + str(episode) + ' Reward: ' + str(np.round(episode_reward,2)))

    pickle.dump(agent.policy_net, open('last_policy.pickle', 'wb'))
    pickle.dump(rewards, open('rewards.pickle', 'wb'))
    pickle.dump(max_angle, open('max_angle.pickle', 'wb'))
    pickle.dump(ave_angle, open('ave_angle.pickle', 'wb'))
    pickle.dump(max_penalties, open('max_penalties.pickle', 'wb'))
    pickle.dump(ave_penalties, open('ave_penalties.pickle', 'wb'))
    pickle.dump(all_traj, open('training_traj.pickle','wb'))
    pickle.dump(ave_w, open('projection_steps.pickle','wb'))

    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Reward vs Episode')
    plt.savefig('rewards.png', dpi=100)
    plt.close()


def run_agent(network_name,seed):
    if network_name == 'pn':
        num_vertex = 2
        policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    elif network_name == 'vpn':
        num_vertex = 2
        policy_net = VertexPolicyNetwork(env=env, obs_dim=obs_dim, num_vertex=num_vertex, hidden_dim=hidden_dim).to(device)
        target_policy_net = VertexPolicyNetwork(env=env, obs_dim=obs_dim, num_vertex=num_vertex, hidden_dim=hidden_dim).to(device)

    torch.manual_seed(seed)
    
    path = os.path.join(parent_dir, 'results/power_system_2' + str(seed) + '/' + network_name)
    
    value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)
    
    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)
    
    agent = DDPG(policy_net=policy_net, value_net=value_net,
                 target_policy_net=target_policy_net, target_value_net=target_value_net)
    
    train_agent(path=path, env=env, agent=agent, seed=seed, num_episodes=num_episodes, num_steps=num_steps)
    
for s in range(10,30):
    run_agent('pn',s)



