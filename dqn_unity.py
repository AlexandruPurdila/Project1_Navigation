import torch      
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from unityagents import UnityEnvironment
from collections import deque

def dqn_unity(env, brain_name, agent, n_episodes=2000, max_t=1000, eps_start=1.0,
              eps_end=0.01, eps_decay=0.995):
  """Deep Q-Learning.
  
  Params
  ======
      env: Unity environment instance
      n_episodes (int): maximum number of training episodes
      max_t (int): maximum number of timesteps per episode
      eps_start (float): starting value of epsilon, for epsilon-greedy action selection
      eps_end (float): minimum value of epsilon
      eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
  """

  scores = []                        # list containing scores from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start                    # initialize epsilon
  for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    for t in range(max_t):
      action = agent.act(state, eps)
      env_info = env.step(action)[brain_name]
      next_state = env_info.vector_observations[0]
      reward = env_info.rewards[0]
      done = env_info.local_done[0]
      agent.step(state, action, reward, next_state, done)
      score+= reward
      state = next_state                    
      if done:
        break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=13.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
      torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
      break
  return scores


if __name__ == '__main__':  
  env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
    
  # get the default brain
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  
  # reset the environment
  env_info = env.reset(train_mode=True)[brain_name]
  
  # number of agents in the environment
  print('Number of agents:', len(env_info.agents))
  
  # number of actions
  action_size = brain.vector_action_space_size
  print('Number of actions:', action_size)
  
  # examine the state space 
  state = env_info.vector_observations[0]
  print('States look like:', state)
  state_size = len(state)
  print('States have length:', state_size)  

  NR_STEPS = 1000
  for ddqn in [False]:
    for dueling in [False]:
      for prioritized_replay in [False]:
        print('Using:')
        print(' * DDNQ: ',  ddqn)
        print(' * DUELING: ', dueling)
        print(' * PRIORITIZED_REPLAY: ', prioritized_replay)
        agent = Agent(
          state_size=state_size, #37 for non-visual or 84x84x1 for visual
          action_size=action_size,  #4
          max_steps=NR_STEPS,
          seed=0, 
          prioritized_replay=prioritized_replay, 
          dueling=dueling, 
          ddqn=ddqn
          )
        scores = dqn_unity(
          env=env, 
          brain_name=brain_name, 
          agent=agent,
          max_t=NR_STEPS
          )
  
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        title = 'ddqn' if ddqn else 'dqn'
        title+= '_dueling' if dueling else ''
        title+= '_prioritized experience replay' if prioritized_replay else '_experience replay'
        plt.title(title)
        plt.show()
        
        max_t = 5
        for i in range(5):
          env_info = env.reset(train_mode=False)[brain_name]
          state = env_info.vector_observations[0]
          score = 0
          for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)
            env_info = env_info[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score+= reward
            state = next_state                    
            if done:
              break 
  env.close()
      