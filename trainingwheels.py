from joblessguy import joblessguy
import gym
import numpy as np
from collections import deque
import os

env=gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size=env.action_space.n

batch_size=32
episodes=1500

agent = joblessguy(state_size, action_size)

done=False
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(5000):
        env.render()

        action=agent.act(state)

        next_state, reward, done, info = env.step(action)

        reward = reward if not done else -10

        next_state=np.reshape(next_state,[1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state=next_state

        if done:
            print("Episode: %d/%d, Score: %d, e: %.2f"%(e,episodes,time,agent.epsilon))
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    if e% 100 ==0:
        agent.save("weights_%d.h5"%e)