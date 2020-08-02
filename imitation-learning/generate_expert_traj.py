"""
Based on the implementation of: https://github.com/Khrylx/PyTorch-RL.
"""
import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import count
from utils import *
parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Ant-v2  ", metavar='G',help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',help='path to the expert policy')

parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N', help='maximal number of main iterations (default: 50000)')
parser.add_argument('--traj-count', type=int, default=32, metavar='N', help='number of expert trajectories to save')
parser.add_argument('--render', default=False, metavar='N', help='render env')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
running_state.fix = True
print("State Clip: {}".format(running_state.clip))
expert_traj = []
def main_loop():
    trajectories = 0
    NUM_TRAJ = args.traj_count
    num_steps = 0
    i_episode =  0
    while trajectories < NUM_TRAJ:
        temp_exprt =  []
        state = env.reset()
        state = running_state(state)
        reward_episode = 0
        for t in range(1234567):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            if not is_disc_action:
                action = policy_net(state_var)[0][0].detach().numpy()
            else:
                action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1
            temp_exprt.append(np.hstack([state, action]))
            if args.render:
                env.render()
            if done:
                break
            state = next_state
        for i in temp_exprt:
            expert_traj.append(i)
        print('Episode {}\t reward: {:.2f}'.format(trajectories, reward_episode))
        trajectories+=1
main_loop()
expert_traj = np.stack(expert_traj)
print(expert_traj.shape)
pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
