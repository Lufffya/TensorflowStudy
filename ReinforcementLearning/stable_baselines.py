import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 2
env = DummyVecEnv([lambda: gym.make('CarRacing-v0') for i in range(n_cpu)])
env = VecFrameStack(env, n_stack=4)
model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log='logs/')
model.learn(total_timesteps=10000000)
model.save("CarRacing_ppo")

del model # remove to demonstrate saving and loading

model = PPO2.load("CarRacing_ppo")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
