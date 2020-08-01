"""
This file is identical to the SIL one, but without the critic network, to perform the
Ablation study of Figure 2 and Appendix B.
"""

import argparse
import os
import torch
import math
import gym
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ot_utils import *
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
from utils          import *
from RL.trpo        import trpo_step
from RL.common      import estimate_advantages
from RL.agent       import Agent
from models.mlp_critic      import Value
from models.mlp_policy import Policy
from models.ot_critic import OTCritic
from torch.autograd import Variable
from torch          import nn
import pickle
import random
import seaborn      as sns
import pandas       as pd
import numpy        as np

parser = argparse.ArgumentParser(description='Sinkhorn Imitation Learning (Ablation Sudy)')
parser.add_argument('--env-name',                    default="Ant-v2", metavar='G', help='name of the environment to run')
parser.add_argument('--expert-traj-path',                                      metavar='G', help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,                         help='render the environment')
parser.add_argument('--log-std', type=float,         default=-0.0,             metavar='G', help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float,           default=0.99,             metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float,             default=0.99,             metavar='G', help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float,          default=1e-3,             metavar='G', help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float,   default=3e-4,             metavar='G', help='lr')
parser.add_argument('--num-threads', type=int,       default=4,                metavar='N', help='Threads')
parser.add_argument('--seed', type=int,              default=1,                metavar='N', help='Seed')
parser.add_argument('--min-batch-size', type=int,    default=50000,             metavar='N', help='minimal batch size per TRPO update (default: 50000)')
parser.add_argument('--max-iter-num', type=int,      default=250,              metavar='N', help='maximal number of main iterations (default: 250)')
parser.add_argument('--log-interval', type=int,      default=1,                metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval',type=int,default=10,               metavar='N', help='interval between saving model (default: 0, means don\'t save)')
parser.add_argument('--gpu-index', type=int,         default=0,                metavar='N', help='Index num of GPU to use')
parser.add_argument('--max-kl', type=float,          default=0.1,             metavar='G', help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float,         default=0.1,             metavar='G', help='damping (default: 1e-2)')
parser.add_argument('--expert-samples', type=int,    default=1000,             metavar='G', help='expert sample number (default: 1000)')
parser.add_argument('--wasserstein-p', type=int,     default=1,                metavar='G', help='p value for Wasserstein')
parser.add_argument('--resume-training',             type=tools.str2bool,      nargs='?', const=True, default=False,  help='Resume training ?')
parser.add_argument('--critic-lr', type=float,       default=5e-4,             metavar='G', help='Critic learning rate')
parser.add_argument('--log-actual-sinkhorn',         type=tools.str2bool,      nargs='?', const=True, default=False,  help='Track actual Sinkhorn with normal cosine cost (for eval only)')
parser.add_argument('--dataset-size', type=int,      default=4,                metavar='G', help='Number of trajectories')
parser.add_argument('--use-mean',         type=tools.str2bool,      nargs='?', const=True, default=False,  help='how to sample from policy')
args = parser.parse_args()
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
env = gym.make(args.env_name)
state_dim      = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim     = 1 if is_disc_action else env.action_space.shape[0]
running_reward  = ZFilter((1,), demean=False, clip=10)
print("Seed: {}".format(args.seed))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)
policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
value_net         = Value(state_dim) # Initialise value network. Used  for calculating Advantages for TRPO
critic_net        = OTCritic(state_dim + action_dim) # Initialise OT critic
if args.resume_training:
    policy_net, value_net,  critic_net, running_state, running_reward = pickle.load(open('assets/learned_models/ablation/SIL/{}/{}_SIL_s{}.p'.format(args.dataset_size, args.env_name, args.seed), "rb"))
to_device(device, policy_net, value_net, critic_net)
optimizer_ot      = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)
optimizer_policy  = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value   = torch.optim.Adam(value_net.parameters(),  lr=args.learning_rate)
#OT params

epsilon           = 0.01                  # Entropy regularisation for Optimal Transport
niter             = 1000000000            # Max Sinknhorn iterations
# load trajectory
expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
running_state.fix = True

interval =  args.subsampling
traj = []
idx = 0
dataset = []
offset = 0
print("Dataset dim:", expert_traj.shape)

agent = Agent(env, policy_net, device, mean_action=False,  running_state=running_state, render=args.render, num_threads=args.num_threads,)

def sil_step(batch, i_iter):
    # Get s,a,r of agent interaction with the environment. This is what agent.collect_samples returns in the main method.
    states      = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    next_states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions     = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    masks       = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    to_device(device, policy_net, value_net, critic_net)
    X = torch.cat([states, actions], 1).to(dtype).to(device) # Concatenate s,a pairs of agent
    with torch.no_grad():
        values = value_net(states)
    for _ in range(1):
        sampled_episodes = []
        epis = []
        for pair in range(len(masks)): # Split to episodes
            epis.append(X[pair].cpu().numpy())
            if masks[pair] == 0:
                sampled_episodes.append(epis)
                epis = []
        total_wasserstein = 0       # Keeps track of all Wassersteins for one episode
        rewards           = []      # Logs rewards to update TRPO
        min_wasserstein   = 10e10   # Used for logging at command line
        max_wasserstein   = 0       # Used for logging at command line
        best_trajectory   = None    # Used for logging at command line
        worst_trajectory  = None    # Used for logging at command line
        index             = 0       # Used for logging at command line
        best_idx          = 0       # Used for logging at command line
        worst_idx         = 0       # Used for logging at command line
        per_trajectory_dis= []      # Used for logging at command line
        cost_loss = []
        num_of_samples =  len(sampled_episodes)-1
        threshold      =  num_of_samples-3
        episodic_eval_sinkhorn = []
        for trajectory in sampled_episodes:
            X = torch.tensor(trajectory).to(dtype).to(device)  # Convert trajectory to tensor.
            sample_traj_index  = random.randint(0,(args.dataset_size-1))
            Y = torch.from_numpy(expert_traj[sample_traj_index]).to(dtype).to(device)   # Comment this out  if you do not want to use expert trajectories, but use the below to use direct expert feedback.
            cost_matrix                = cosine_distance(X, Y)                     # Get cost matrix for samples using fixed cosine transport cost.
            transport_plan             = optimal_transport_plan(X, Y, cost_matrix, method='sinkhorn_gpu') # Getting optimal coupling
            per_sample_costs           = torch.diag(torch.mm(transport_plan, cost_matrix.T)) # Get diagonals W = MC^T, where M is the optimal transport map and C the cost matrix
            distance                   = torch.sum(per_sample_costs)                         # Calculate Wasserstein by summing diagonals, i.e., W=Trace[MC^T]
            wasserstein_distance       = -(distance)                                         # Assign -wasserstein in order to gradient descent to maximise if using adversary for training.
            per_trajectory_dis.append(distance.detach().cpu().numpy())                       # Keep track of all Wasserstein distances in one sample.
            episodic_eval_sinkhorn.append(distance.item())
            if distance < min_wasserstein and index != (len(sampled_episodes)):              # Keep track of best trajectory based on Wasserstein distance
                min_wasserstein = distance
                best_trajectory = X
                best_idx        = index
            if distance > max_wasserstein and index != (len(sampled_episodes)): # Keep track of worst trajectory based on Wasserstein distance
                max_wasserstein  = distance
                worst_trajectory = X
                worst_idx        = index
            index  += 1
            counter = 0
            survival_bonus = 4 / X.shape[0]
            for per_sample_cost in per_sample_costs: # Add rewards
                with torch.no_grad():
                    temp_r =   -2 * per_sample_cost +  survival_bonus
                    temp_r.unsqueeze_(0)
                    temp_r = running_reward(temp_r.cpu())
                    rewards.append(temp_r)
                    counter += 1
            total_wasserstein += distance
            torch.cuda.empty_cache()
    with torch.no_grad():
        rewards     =  torch.tensor(rewards)
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device) # Get Advantages for TRPO
    torch.cuda.empty_cache()
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg) # Update policy
    return (total_wasserstein**2)**(1/2), episodic_eval_sinkhorn, len(sampled_episodes), min_wasserstein, best_trajectory, best_idx, max_wasserstein, worst_trajectory, worst_idx,  per_trajectory_dis
def ablated_sil():

    print("------------ Sinkhorn based Imitation Learning -------------")
    print("---Parameters:----\n")
    print("KL: {}"                           .format(args.max_kl))
    print("Damping: {}"                      .format(args.damping))
    print("Value Function Regularisation: {}".format(args.l2_reg))
    print("Critic Learning Rate: {}"         .format(args.critic_lr))
    print("γ: {}"                            .format(args.gamma))
    print("τ: {}"                             .format(args.tau))
    W_loss = []
    R_avg  = []
    R_max  = []
    R_min  = []
    episodic_rewards = []
    sinkhorn_log =  []
    if args.resume_training ==True:
        R_avg  =  pickle.load(open('experiment-logs-sil/ablation/{}/{}/avg_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        R_max  =  pickle.load(open('experiment-logs-sil/ablation/{}/{}/max_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        R_min  =  pickle.load(open('experiment-logs-sil/ablation/{}/{}/min_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        sinkhorn_log =  pickle.load(open('experiment-logs-sil/ablation/{}/{}/skh_seed{}.l'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        episodic_rewards =  pickle.load(open('experiment-logs-sil/ablation/{}/{}/episodic_rewards_seed{}.l'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        args.max_iter_num  =  args.max_iter_num - len(R_avg)
        print("Episodes left: {}".format(args.max_iter_num))
        input("continue ?")
    episode= []
    loss   =10e3
    for i_iter in range(args.max_iter_num):
        t0 = time.time()
        batch, log = agent.collect_samples(args.min_batch_size, randomise = False)
        loss, eval_sinkhorn_per_episode,  sampled_episodes, min_loss, best_trajectory, best_idx, max_loss, worst_trajectory, worst_idx, per_trajectory_dis  = sil_step(batch, i_iter)
        sinkhorn_log.append(eval_sinkhorn_per_episode)
        bestR            = log['episodic_rewards'][best_idx]
        worstR           = log['episodic_rewards'][worst_idx]
        episode.append(i_iter)
        W_loss.append(eval_sinkhorn_per_episode)
        R_avg.append(log['avg_reward'])
        R_max.append(log['max_reward'])
        R_min.append(log['min_reward'])
        episodic_rws = log['episodic_rewards']
        episodic_rewards.append(episodic_rws)
        if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, critic_net)
            pickle.dump((policy_net, value_net, critic_net, running_state, running_reward), open(os.path.join(assets_dir(), 'learned_models/ablation/SIL/{}/{}_SIL_s{}.p'.format(args.dataset_size, args.env_name, args.seed)), 'wb'))
            to_device(device, policy_net, value_net)
            pickle.dump(R_avg, open('experiment-logs-sil/ablation/{}/{}/avg_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(R_max, open('experiment-logs-sil/ablation/{}/{}/max_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(R_min, open('experiment-logs-sil/ablation/{}/{}/min_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(sinkhorn_log, open('experiment-logs-sil/ablation/{}/{}/skh_seed{}.l'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(episodic_rewards, open('experiment-logs-sil/ablation/{}/{}/episodic_rewards_seed{}.l'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
        torch.cuda.empty_cache()
        t1 = time.time()
        if i_iter % args.log_interval == 0:
            print('{}\tTime: {:.4f}\tR_max {:.2f}\tR_avg {:.2f}\tR_min {:.2f}\tAdv. Sinkhorn {:.4f}\tActual Sinkhorn {:.4}\tSampled Episodes {}\tBT:[Pairs: {} | S: {:.4} | R: {:.2f}]\tWT:[Pairs: {} | W: {:.4} | R: {:.2f}]'.format(
                i_iter, t1-t0, log['max_reward'], log['avg_reward'], log['min_reward'], loss.item(), sum(eval_sinkhorn_per_episode)/sampled_episodes, sampled_episodes, best_trajectory.shape[0], min_loss, bestR, worst_trajectory.shape[0], max_loss, worstR))
ablated_sil()
