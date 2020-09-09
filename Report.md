# 1. Project Goal
The goal of this project is to build a RL agent capable of achieving a score of 13 (or more) collected bananas over 100 consecutive episodes

# 2. Report Objective
In this report I briefly summarize the proposed solution that trains a RL agent to navigate and collect bananas inside Unity's Banana Collector environment.

# 3. Solution overview
The current solution consists of a configurable Agent that:
- can be trained using (standard) DQN or Double DQN mode
- can be trained using (standard) Replay Buffer or Prioritized Replay Buffer mode
- can be trained using dueling mode

# 4. Agent "brain"
The Agent "brain" consist of a small footprint PyTorch model composed of just two hidden linear layers (64 units each), each followed by a ReLU activation. The model continues with two separate linear layers to provide the dueling functionality, each layer containing 64 units.

# 5. Replay Buffer & Prioritized Replay Buffer
The Agent experience can be stored and reused either from a standard Replay Buffer or a prioritized one. The 

# 6. Training loop
The Agent is trained using 'dqn_unity' with the following parameters:
- `DQN mode`
- `Replay Buffer mode`
- `no Dueling`
- `eps_start=1.0`
- `eps_end=0.01` 
- `eps_decay=0.995`
- `gamma=0.99`
- `batch_size=64`
- `lr=5e-4`

# 7. The results
The Agent solves the environment in 494 episodes, with an average score of 13.04:

![alt text](https://github.com/AlexandruPurdila/Project1_Navigation/blob/master/agent_score.png)

# 8. Further work
Giving the fact that the Agent is configurable, further testing will be made in the form of a grid search stage in order to compare the behaviour of the model when allowing more complex setups (like DDQN, dueling, prioritized experience replay, etc.).
