import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from   itertools import count
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
import statistics
from ot_utils import *

parser = argparse.ArgumentParser(description='Test learner policy after imitation')
parser.add_argument('--env-name',               default="Ant-v2", metavar='G', help='name of the environment to run')
parser.add_argument('--model-path',             metavar='G', help='name of the expert model')
parser.add_argument('--render',                 action='store_true', default=False, help='Render MuJoCo')
parser.add_argument('--seed',                   type=int, default=1, metavar='N',help='Random Seed')
parser.add_argument('--max-expert-state-num',   type=int, default=6000, metavar='N',help=' ')
parser.add_argument('--episodes',               type=int, default=50, metavar='N', help='Episodes to test on (default: 50 as in paper)')
parser.add_argument('--dataset-size',           type=int, default=4, metavar='N', help='Number of expert demonstrations provided')
parser.add_argument('--expert-traj-path',       metavar='G', help='name of the expert model')
parser.add_argument('--method',                 metavar='G', default='sil',  help='Method to test learner policyw')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env   = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
print(args.model_path)
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=-0.00)

if args.method  ==  ('gail' or 'airl'):
    policy_net,_,_,running_state = pickle.load(open(args.model_path, "rb"))
else:
    policy_net,_,_,running_state,_ = pickle.load(open(args.model_path, "rb"))
subsampled_expert_traj , _ = pickle.load(open(args.expert_traj_path, "rb"))
print(running_state.clip)
running_state.fix = True

def testpolicy():
    avr_r = 0
    eps  = 0
    num_steps = 0
    max_r = -10e10
    min_r = 10e10
    rws=[]
    episodic_eval_sinkhorn=[]
    for i_episode in range(args.episodes):
        episode = []
        state = env.reset()
        state = running_state(state)
        reward_episode = 0
        for t in range(2000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            if is_disc_action:
                action = np.argmax(policy_net(state_var)[0].detach().numpy())
            else:
                action = policy_net(state_var)[0][0].detach().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1
            episode.append(np.hstack([state, action]))
            if args.render:
                env.render()
            if done:
                break
            state = next_state
        sample_traj_index  = random.randint(0,(args.dataset_size-1))
        X = torch.tensor(episode)
        Y_eval = torch.from_numpy(subsampled_expert_traj[sample_traj_index]).to(dtype).to(device)
        evaluation_cost_matrix = cosine_distance(X, Y_eval)
        evaluation_transport_plan = optimal_transport_plan(X, Y_eval, evaluation_cost_matrix, method='sinkhorn_gpu')
        eval_wasserstein_distance = torch.sum(torch.diag(torch.mm(evaluation_transport_plan, evaluation_cost_matrix.T)))
        torch.cuda.empty_cache()
        print('Episode {}\t reward: {:.2f}\t Sinkhorn: {:.2f}'.format(i_episode, reward_episode, eval_wasserstein_distance.item()))
        if reward_episode > max_r:
            max_r  = reward_episode
        if  reward_episode < min_r:
            min_r = reward_episode
        avr_r += reward_episode
        eps+=1
        episodic_eval_sinkhorn.append(eval_wasserstein_distance.item())
        rws.append(reward_episode)
    print("Average Reward: {}\t Max Reward: {}\t Min R: {}\t Steps: {}\t Stdev: {}".format(avr_r/eps, max_r, min_r, num_steps, statistics.stdev(rws)))
    print("Average Sinkhorn: {}\t Stdev: {}".format(sum(episodic_eval_sinkhorn)/len(episodic_eval_sinkhorn), statistics.stdev(episodic_eval_sinkhorn)))

testpolicy()


