import gym

env = gym.make('Marvin-v0')

for _ in range(1000):
	env.render()
	env.step(env.action_space.sample())
