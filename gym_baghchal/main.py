import gym
env = gym.make('gym_baghchal/Baghchal-v0')
obs = env.reset()
done = False
while not done:
    action = (0, 0) # Replace with your own action
    obs, reward, done, info = env.step(action)
    env.render(mode='human') # Replace with 'ascii' or 'pygame' if desired
