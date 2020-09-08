import random
import numpy as np
import torch
from collections import deque, namedtuple


class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed):
    """Initialize a ReplayBuffer object.

    Params
    ======
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
    return
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)
    if len(experiences[0].state.shape) == 3:
      states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
      next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
    else:
      states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
      next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
    #endif
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)