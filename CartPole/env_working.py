import gym

env = gym.envs.make("CartPole-v0")
env.reset()

done = False

while not done:
	action = env.action_space.sample()
	obs, rew, done, _ = env.step(action)
	print(action, obs, rew, done)
