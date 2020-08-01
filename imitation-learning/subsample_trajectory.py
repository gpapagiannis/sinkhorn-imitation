import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
import random

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Ant-v2", metavar='G', help='')
parser.add_argument('--traj-path', metavar='G',help='')
parser.add_argument('--number-of-traj', type=int, default=4,help='')
parser.add_argument('--episode-size', type=int, default=1000, help='')
parser.add_argument('--subsampling', type=int, default=20, help='')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
args = parser.parse_args()
dtype = torch.float64
torch.set_default_dtype(dtype)
env   = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
def subsample():
    expert_traj, running_state = pickle.load(open("assets/expert_traj/{}_expert_traj.p".format(args.env_name), "rb"))
    expert_traj = expert_traj[0:(args.number_of_traj * args.episode_size)]
    interval =  args.subsampling
    idx          = 0
    dataset      = []
    offset       = 1
    episode      = []
    trajectories = []
    subsampled_trajectories = []
    for i in range(len(expert_traj)):
        episode.append(expert_traj[i])
        if (i+1) % args.episode_size == 0:
            trajectories.append(episode)
            episode=[]
    subsampled_episode=[]
    for traj in trajectories:
        for i in range(len(traj)):
            if (i-offset+1) % interval == 0:
                subsampled_episode.append(traj[i])
        print(len(subsampled_episode))
        subsampled_trajectories.append(subsampled_episode)
        subsampled_episode = []
        offset=random.randint(0, args.subsampling)
    expert_traj = np.asarray(subsampled_trajectories)
    pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'subsampled_expert_traj/{}/{}'.format(args.number_of_traj, args.env_name)), 'wb'))

subsample()
