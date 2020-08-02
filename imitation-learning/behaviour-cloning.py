import argparse
import math
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.cm     as cm
import matplotlib.pyplot as plt
from utils                  import *
from RL.agent               import Agent
from models.mlp_policy      import Policy
from models.mlp_critic      import Value
from torch.autograd         import Variable
from torch                  import nn
import pickle
import random
import seaborn      as sns
import pandas       as pd
import numpy        as np

parser = argparse.ArgumentParser(description='Behavioural Cloning')
parser.add_argument('--env-name',                    default="Ant-v2", metavar='G', help='name of the environment to run')
parser.add_argument('--expert-traj-path',                                      metavar='G', help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,                         help='render the environment')
parser.add_argument('--learning-rate', type=float,   default=3e-4,             metavar='G', help='gae')
parser.add_argument('--gpu-index', type=int,         default=0,                metavar='N', help='Index num of GPU to use')
parser.add_argument('--expert-samples', type=int,    default=1000,             metavar='G', help='expert sample number')
parser.add_argument('--subsampling', type=int,       default=1,                metavar='G', help="Subsampling of expert's demonstration")
parser.add_argument('--dataset-size', type=int,      default=4,                metavar='G', help="Dataset Size")
parser.add_argument('--seed', type=int,      default=4,                metavar='G', help="Dataset Size")
parser.add_argument('--num-threads', type=int,      default=4,                metavar='G', help="Dataset Size")
parser.add_argument('--iterations', type=int,      default=100,                metavar='G', help="Dataset Size")


args = parser.parse_args()
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
device_gpu = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
env = gym.make(args.env_name)
print("Seed: {}".format(args.seed))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

# load trajectory
subsampled_expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
running_state.fix = True
print(running_state.clip)
print(subsampled_expert_traj.shape)
expert_traj = []
for t in subsampled_expert_traj:
    for t_i in t:
        expert_traj.append(t_i)
expert_traj =  np.asarray(expert_traj)
state_dim = env.observation_space.shape[0]
action_dim =  env.action_space.shape[0]

policy_net = Policy(state_dim, env.action_space.shape[0])
to_device(device, policy_net)
policy_optimiser  = torch.optim.Adam(policy_net.parameters(), lr=0.0001,  betas=(0.0, 0.999))

agent = Agent(env, policy_net, device, mean_action=False,  running_state=running_state, render=args.render, num_threads=args.num_threads,)

def bc(iterations = args.iterations):

    et     = torch.from_numpy(expert_traj).to(dtype).to(device)
    states = et[::, 0:state_dim]
    acts   = et[::, state_dim:(state_dim+action_dim)]
    states = states.clone().cpu()
    print(states.size())
    batch_size = 1024
    
    for k in range(iterations): # Train BC
        permutation = torch.randperm(states.size()[0])
        for i in range(0,states.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = states[indices, ::], acts[indices, ::] 
            policy_optimiser.zero_grad()
            log_prob = policy_net.get_log_prob(batch_x, batch_y)
            loss = -1.0 * log_prob.mean()
            loss.backward()
            policy_optimiser.step()
        if k % 100 ==  0:
            print('{}\t Loss: {:.4}'.format(k, loss.item()))

def main_loop():
    bc()
    to_device(torch.device('cpu'), policy_net)
    pickle.dump((policy_net, running_state), open(os.path.join(assets_dir(), 'learned_models/BC/{}/{}_BC_s{}.p'.format( args.dataset_size, args.env_name, args.seed)), 'wb'))
    to_device(device, policy_net)

main_loop()
