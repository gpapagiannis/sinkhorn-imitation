import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from RL.trpo import  trpo_step
from RL.common import estimate_advantages
from RL.agent import Agent

from ot_utils import optimal_transport_plan, cosine_distance

parser = argparse.ArgumentParser(description='Generative Adversarial Imitation Learning')
parser.add_argument('--env-name',                    default="Ant-v2", metavar='G', help='name of the environment to run')
parser.add_argument('--expert-traj-path',                                      metavar='G', help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,                         help='render the environment')
parser.add_argument('--log-std', type=float,         default=-0.0,             metavar='G', help='log std for the policy ')
parser.add_argument('--gamma', type=float,           default=0.99,             metavar='G', help='discount factor')
parser.add_argument('--tau', type=float,             default=0.99,             metavar='G', help='gae')
parser.add_argument('--l2-reg', type=float,          default=1e-3,             metavar='G', help='l2 regularization regression')
parser.add_argument('--learning-rate', type=float,   default=3e-4,             metavar='G', help='lr')
parser.add_argument('--num-threads', type=int,       default=4,                metavar='N', help='Threads')
parser.add_argument('--seed', type=int,              default=1,                metavar='N', help='Seed')
parser.add_argument('--min-batch-size', type=int,    default=50000,             metavar='N', help='minimal batch size per TRPO update ')
parser.add_argument('--max-iter-num', type=int,      default=250,              metavar='N', help='maximal number of main iterations')
parser.add_argument('--log-interval', type=int,      default=1,                metavar='N', help='interval between training status logs')
parser.add_argument('--save-model-interval',type=int,default=10,               metavar='N', help='interval between saving model')
parser.add_argument('--gpu-index', type=int,         default=0,                metavar='N', help='Index num of GPU to use')
parser.add_argument('--max-kl', type=float,          default=0.1,             metavar='G', help='max kl value')
parser.add_argument('--damping', type=float,         default=0.1,             metavar='G', help='damping')
parser.add_argument('--expert-samples', type=int,    default=1000,             metavar='G', help='expert sample number')
parser.add_argument('--wasserstein-p', type=int,     default=1,                metavar='G', help='p value for Wasserstein')
parser.add_argument('--resume-training',             type=tools.str2bool,      nargs='?', const=True, default=False,  help='Resume training ?')
parser.add_argument('--critic-lr', type=float,       default=5e-4,             metavar='G', help='Critic learning rate')
parser.add_argument('--dataset-size', type=int,      default=4,                metavar='G', help='Number of trajectories')
parser.add_argument('--use-mean',         type=tools.str2bool,      nargs='?', const=True, default=False,  help='how to sample from policy')
parser.add_argument('--batch', type=int,             default=1024,                        metavar='G', help='discriminator batch size')
parser.add_argument('--momentum', type=float,             default=0.0,                        metavar='G', help='discriminator batch size')

args = parser.parse_args()
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
value_net = Value(state_dim)
discriminator = Discriminator(state_dim + action_dim)
discrim_criterion = nn.BCELoss()
to_device(device, policy_net, value_net, discriminator, discrim_criterion)
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))
if args.resume_training:
    policy_net, value_net, discriminator, running_state = pickle.load(open('assets/learned_models/GAIL/{}/{}_GAIL_s{}.p'.format(args.dataset_size, args.env_name, args.seed), "rb"))
# load trajectory and concatenate all demonstration sets - this is done because subsampled trajectories
# are saved in the form (traj_num, samples, dim) to fit in SIL's algorithm (1). Here we put them all together for GAIL.
subsampled_expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
running_state.fix = True
print(subsampled_expert_traj.shape)
expert_traj = []
for t in subsampled_expert_traj:
    for t_i in t:
        expert_traj.append(t_i)
expert_traj =  np.asarray(expert_traj)
agent = Agent(env, policy_net, device, custom_reward=None,
              running_state=running_state, render=args.render, num_threads=args.num_threads)
def gail_step(batch, i_iter):
    to_device(device, policy_net, value_net, discriminator)
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
    X = torch.cat([states, actions], 1).to(dtype).to(device) # Concatenate s,a pairs of agent
    Y = torch.from_numpy(expert_traj).to(dtype).to(device)
    rewards = []
    rs  = discriminator(X).detach().clone()
    for r in rs:
        rewards.append(-math.log(r.item())) #gail reward
    rewards  = torch.tensor(rewards)
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
    sampled_episodes = []
    epis = []
    for pair in range(len(masks)): # Split to episodes for evaluation only
        epis.append(X[pair].cpu().numpy())
        if masks[pair] == 0:
            sampled_episodes.append(epis)
            epis = []
    batch_size = args.batch
    for ep in range(1):
        permutation = torch.randperm(X.size()[0])
        for i in range(0,X.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X[indices, ::]
            learner_samples_disc = discriminator(batch_x)
            expert_samples_disc = discriminator(Y)
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(learner_samples_disc, ones((batch_x.shape[0], 1), device=device)) + \
                discrim_criterion(expert_samples_disc, zeros((expert_traj.shape[0], 1), device=device))
            discrim_loss.backward()
            optimizer_discrim.step()
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg) # Update policy (TRPO from: https://github.com/Khrylx/PyTorch-RL)
    return len(sampled_episodes)

def gail():
    R_avg  = []
    R_max  = []
    R_min  = []
    episodic_rewards = []
    if args.resume_training ==True:
        R_avg  =  pickle.load(open('experiment-logs-gail/{}/{}/avg_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        R_max  =  pickle.load(open('experiment-logs-gail/{}/{}/max_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        R_min  =  pickle.load(open('experiment-logs-gail/{}/{}/min_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        episodic_rewards =  pickle.load(open('experiment-logs-gail/{}/{}/episodic_rewards_seed{}.l'.format(args.env_name, args.dataset_size, args.seed), "rb"))
        args.max_iter_num  =  args.max_iter_num - len(R_avg)
        print("Episodes left: {}".format(args.max_iter_num))
        input("continue ?")
    for i_iter in range(args.max_iter_num):
        discriminator.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size)
        discriminator.to(device)
        t0 = time.time()
        num_of_samples =  gail_step(batch, i_iter)
        t1 = time.time()
        if i_iter % args.log_interval == 0:
            print('{}\tSamples (s) {:.4f}\tUpdate  (s) {:.4f}\tMinimum R {:.2f}\tAverage R {:.2f}\t Maximum R {:.2f}\t'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['avg_reward'], log['max_reward']))
        R_avg.append(log['avg_reward'])
        R_max.append(log['max_reward'])
        R_min.append(log['min_reward'])
        episodic_rewards.append(log['episodic_rewards'])
        if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discriminator)
            pickle.dump((policy_net, value_net, discriminator, running_state), open(os.path.join(assets_dir(), 'learned_models/GAIL/{}/{}_GAIL_s{}.p'.format(args.dataset_size, args.env_name, args.seed)), 'wb'))
            to_device(device, policy_net, value_net)
            pickle.dump(R_avg, open('experiment-logs-gail/{}/{}/avg_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(R_max, open('experiment-logs-gail/{}/{}/max_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(R_min, open('experiment-logs-gail/{}/{}/min_seed{}.r'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
            pickle.dump(episodic_rewards, open('experiment-logs-gail/{}/{}/episodic_rewards_seed{}.l'.format(args.env_name, args.dataset_size, args.seed), 'wb'))
        torch.cuda.empty_cache()
gail()
