import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
  """Actor (Policy) Model."""

  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dueling=False):
    """Initialize parameters and build model.
    Params
    ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
    """
    super(QNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.dueling = dueling
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    
    self.fc_adv = nn.Linear(fc2_units, action_size)
    self.fc_val = nn.Linear(fc2_units, 1)
    return

  def forward(self, state):
    """Build a network that maps state -> action values."""
    th_x = state
    th_x = F.relu(self.fc1(th_x))
    th_x = F.relu(self.fc2(th_x))
    th_adv = self.fc_adv(th_x)
    
    th_res = th_adv
    if self.dueling:
      th_val = self.fc_val(th_x)
      th_adv_mean = torch.mean(th_adv, dim=1, keepdim=True)
      th_duel = th_val + th_adv - th_adv_mean
      th_res = th_duel
    #endif
    return th_res