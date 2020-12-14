import os
import matplotlib.pyplot as plt
import gym
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


env = gym.make("CarRacing-v0")


for i_episode in range(20):
    observation = env.reset()

    for t in range(100):

        # plt.imshow(observation)
        # plt.show()

        env.render()

        print(observation)

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
