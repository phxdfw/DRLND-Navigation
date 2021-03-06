import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    fc1_units = 48
    fc2_units = 64
    fc3_units = 48
    fc4_units = 36
        
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, QNetwork.fc1_units)
        self.fc2 = nn.Linear(QNetwork.fc1_units, QNetwork.fc2_units)
        self.fc3 = nn.Linear(QNetwork.fc2_units, QNetwork.fc3_units)
        self.fc4 = nn.Linear(QNetwork.fc3_units, QNetwork.fc4_units)
        self.fc5 = nn.Linear(QNetwork.fc4_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
