import gym
from joblessguy import joblessguy
import numpy as np

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size=env.action_space.n

agent = joblessguy(state_size, action_size)
agent.load('weights_1500.h5')

done=False


for i in range(100):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(5000):
        env.render()
        act_values = agent.model.predict(state)
        action=np.argmax(act_values[0])
        next_state, reward, done, info = env.step(action)
        state= next_state
        state = np.reshape(state, [1, state_size])

        if done:
            break
