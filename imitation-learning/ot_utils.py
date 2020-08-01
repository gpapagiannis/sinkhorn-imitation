import argparse
import math
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import  ot.gpu
#import  ot
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
dtype = torch.float64
torch.set_default_dtype(dtype)
device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epsilon           = 0.01                  # Entropy regularisation for Optimal Transport
niter             = 1000000000            # Max Sinknhorn iterations

"""
Return optimal transport map
"""
def optimal_transport_plan(X, Y, cost_matrix, method='emd'):
    X_pot = Variable(torch.ones(X.shape[0])* (1/X.shape[0]), requires_grad=False) # Create uniform distr. for samples of agent
    Y_pot = Variable(torch.ones(Y.shape[0])* (1/Y.shape[0]), requires_grad=False) # Create uniform distr. for samples of expert
    X_pot = X_pot.cpu().numpy() # To numpy for POT library
    Y_pot = Y_pot.cpu().numpy() # To numpy for POT library
    c_m_clone = Variable(cost_matrix.data.clone(), requires_grad=False) # In order to get the cost matrix on GPU, first determine using tensors the convert to numpy for POT library
    c_m       = c_m_clone.cpu().numpy() # Clone cost matrix to numpy for POT library
    del c_m_clone; torch.cuda.empty_cache()
    if method == 'sinkhorn_gpu':
        transport_plan               = ot.gpu.sinkhorn(X_pot,Y_pot,c_m, epsilon, numItermax=niter)     # (GPU) Get the transport plan for regularized OT
    elif method  == 'sinkhorn':
        transport_plan               = ot.sinkhorn(X_pot,Y_pot,c_m,epsilon, numItermax=niter)          # (CPU) Get the transport plan for regularized OT
    else:
        transport_plan               = ot.emd(X_pot,Y_pot,c_m, numItermax=niter)                       # (CPU) Get the transport plan for OT with no regularisation
    transport_plan               = torch.from_numpy(transport_plan).to(dtype).to(device)
    transport_plan.requires_grad = False
    del X_pot; del Y_pot; torch.cuda.empty_cache()
    return transport_plan

"""
-------------------------------------------------------------
A  more efficient way to determine the Wasserstein distance.
For now only good when using on GPU for memory efficiency as
it does not allow for fast adversarial training.
-------------------------------------------------------------
"""
def calculate_wassertstein(tp, cm):
    tp.to(device)
    cm.to(device)
    s = cm.shape[0]
    cm =  cm.T #Transpose cost matric for convention
    wasserstein_costs = []
    cumulative_cost = Variable(torch.zeros([1,1]), requires_grad=True)
    for i in range(s):
        cost = torch.dot(tp[i,:], cm[:,i])
        cumulative_cost = cumulative_cost + cost
        wasserstein_costs.append(cost)
    return wasserstein_costs, cumulative_cost

"""
Normal Euclidean distance.
"""
def W_p(x, y, p=2):
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)**(1/p)
    return C

"""
Normal cosine distance.
"""
def cosine_distance(x, y):
    x = x.to(device)
    y = y.to(device)
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1).to(device)
    y_norm = torch.norm(y, p=2, dim=1).to(device)
    x_n  = x_norm.unsqueeze(1).to(device)
    y_n  = y_norm.unsqueeze(1).to(device)
    norms = torch.mm(x_n, y_n.T).to(device)
    C = (1-C/norms)
    return C.to(device)
"""
Euclidean distance with critic network.
As expected, it gets very, very large.
"""
def critic_W_p(x, y,cost_fn, p=2):
    x = cost_fn(x)
    y = cost_fn(y)
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1) ** (1/p)

    return C
"""
Cosine distance, with critic network.
"""
def cosine_critic(x, y, cost_fn, scale_cost=1):
    x = x.to(device)
    y = y.to(device)
    x = cost_fn(x).to(device)
    y = cost_fn(y).to(device)
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1).to(device)
    y_norm = torch.norm(y, p=2, dim=1).to(device)
    x_n  = x_norm.unsqueeze(1).to(device)
    y_n  = y_norm.unsqueeze(1).to(device)
    norms = torch.mm(x_n, y_n.T).to(device)
    C = 1-C/norms
    C  = scale_cost*C
    return C.to(device)
