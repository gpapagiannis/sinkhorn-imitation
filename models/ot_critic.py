import torch.nn as nn
import torch


class OTCritic(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        
        self.logic = nn.Linear(last_dim, 30)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        return self.logic(x)
    
    


