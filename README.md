# Imitation Learning with Sinkhorn Distances

Code for the experiments in the work: Imitation Learning with Sinkhorn Distances.

## Prerequisites 
* A version of python 3
* To run the experiments, first install the MuJoco simulator (https://github.com/openai/mujoco-py) and OpenAIGym (https://gym.openai.com/docs/).
* Other project specific requirements can be installed directly via:
```
pip install -r requirements.txt
```
* We recommend running the experiments on a GPU of memory at least 16GB, to efficienlty obtain the Sinkhorn distance.

## Sinkhorn Imitation Learning (SIL)


## Steps to perform imitation learning (Example)
This example shows the steps to perform imitation learning using SIL for the Ant-v2 environment with 16 expert trajectories. The same process can be repeated for other experiments (Hopper-v2, Walker2d-v2, HalfCheetah-v2 and Humanoid-v2) and algorithms (GAIL and AIRL)


### Step 1: Train RL agent to obtain expert performance
```
python agent/trpo_gym.py  --env-name Ant-v2
```
### Step 2: Generate expert trajectory using the expert policy
```
python imitation-learning/generate_expert_traj.py --env-name Ant-v2 --model-path assets/learned_models/expert-policies/Ant-v2_trpo.p --traj-count 16
```
### Step 3: Subsample the expert trajectory
```
python imitation-learning/subsample_trajectory.py --env-name  Ant-v2 --traj-path assets/expert_traj/Ant-v2_expert_traj.p --number-of-traj 16
```
### Step 4: Train Sinkhorn Imitation Learning
The results are saved with the seed number in the end in order to be able to keep track performance amongst difference seeds. To change the critic network's output edit the file: ```./models/ot_critic.py```.
```
python imitation-learning/SIL.py --env-name Ant-v2 --expert-traj-path assets/subsampled_expert_traj/16/Ant-v2 --gamma 0.99 --tau 0.97 --min-batch-size 50000 --seed 123 --max-iter-num 250 --log-actual-sinkhorn True --critic-lr .0005 --dataset-size 16
```
### Step 5: Test SIL performance
Testing the policy will return performance in terms of the reward metric and the Sinkhorn distance using a fixed cosine transport cost
```
python imitation-learning/test_policy.py --env-name Ant-v2 --model-path assets/learned_models/SIL/16/Ant-v2_SIL_s123.p --expert-traj-path assets/subsampled_expert_traj/16/Ant-v2 --dataset-size 16 --episodes 50 --method sil
```

### References
* A large amount of the Reinforcement Learning code and folder structure was based on the implementation of Ye Yuan [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL/blob/master/README.md) 
* [Generative Adversarial Imitation Learning (GAIL)](https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning)
* [Adversarial Inverse Reinforcement Learning (AIRL)](https://openreview.net/pdf?id=rkHywl-A-)
* [Inverse RL by Justin Fu Implementation](https://github.com/justinjfu/inverse_rl)



