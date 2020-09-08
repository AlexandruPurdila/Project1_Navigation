import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from model import QNetwork
from replay_buffer import ReplayBuffer
from per import PrioritizedReplayBuffer


class Agent:
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, max_steps, seed, 
               prioritized_replay, dueling, ddqn,
               buffer_size=int(1e5), batch_size=64, gamma=0.99,
               tau=1e-3, lr=5e-4, update_every=4,
               prioritized_replay_alpha=0.6, prioritized_replay_eps=1e-6,
               prioritized_replay_beta0=0.4):
    """Initialize an Agent object.
    
    Params
    ======
        state_size (int)            : dimension of each state
        action_size (int)           : dimension of each action
        seed (int)                  : random seed
        prioritized_replay (bool)   : use prioritized replay or standard replay buffer
        dueling (bool)              : use dueling agent
        ddqn (bool)                 : use ddqn agent
        buffer_size (int)           : replay buffer size
        batch_size (int)            : minibatch size
        gamma (float)               : discount factor
        tau (float)                 : for soft update of target parameters
        lr (float)                  : learning rate 
        update_every (int)          : how often to update the network
    """
    self.state_size = state_size
    self.action_size = action_size
    self.max_steps = max_steps
    self.seed = random.seed(seed)
    self.prioritized_replay = prioritized_replay
    self.dueling = dueling
    self.ddqn = ddqn
    self.buffer_size = buffer_size
    self.batch_size=batch_size
    self.gamma = gamma
    self.tau = tau
    self.lr = lr
    self.update_every = update_every
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed, dueling=self.dueling).to(self.device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed, dueling=self.dueling).to(self.device)
    
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

    # Replay memory
    if not self.prioritized_replay:
      self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
    else:
      self.memory = PrioritizedReplayBuffer(size=self.buffer_size, alpha=self.prioritized_replay_alpha)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    return
    
  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory        
    # Common interface for both Prioritized and Standard Replay
    self.memory.add(state, action, reward, next_state, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.update_every
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > self.batch_size:
        if not self.prioritized_replay:
          experiences = self.memory.sample()
        else:
          fraction = self.prioritized_replay_beta0 * self.t_step / self.max_steps
          beta = self.prioritized_replay_beta0 + fraction * (1 - self.prioritized_replay_beta0)
          experiences = self.memory.sample(batch_size=self.batch_size, beta=beta)
        #endif
        self.learn(experiences, self.gamma)
      #endif
    #endif
    return

  def act(self, state, eps=0.):
    """Returns actions for given state as per current policy.
    
    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    self.qnetwork_local.eval()
    with torch.no_grad():
        action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy()).astype(int)
    else:
      return random.choice(np.arange(self.action_size)).astype(int)
    return

  def learn(self, experiences, gamma, is_weights=None, idxes=None):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """
    if not self.prioritized_replay:
      states, actions, rewards, next_states, dones = experiences
    else:
      states, actions, rewards, next_states, dones, is_weights, batch_idxes = experiences
      states = torch.from_numpy(np.vstack(states)).float().to(self.device)
      actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
      rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
      next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
      dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
      is_weights = torch.from_numpy(np.vstack(is_weights)).float().to(self.device)
    
    if not self.ddqn:
      # Get max predicted Q values (for next states) from target model
      Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
      # Compute Q targets for current states 
    else:
      #Action selection (select action using local network)
      action_targets_sel = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
      #Action evaluation (evaluate action using target)
      Q_targets_next = self.qnetwork_target(next_states).gather(1, action_targets_sel)
    #endif

    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)
    
    if self.prioritized_replay:
      td_errors = Q_expected - Q_targets
      new_priorities = torch.abs(td_errors).detach().cpu().numpy() + self.prioritized_replay_eps
      self.memory.update_priorities(idxes=batch_idxes, priorities=new_priorities)
    
    # Compute loss
    if not self.prioritized_replay:
      loss = F.mse_loss(Q_expected, Q_targets).mean()
    else:
      loss = (F.mse_loss(Q_expected, Q_targets) * is_weights).mean()
    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    return

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    return