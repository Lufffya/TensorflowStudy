from tensorflow.keras import layers
from tensorflow import keras
import gym
import numpy as np
import tensorflow as tf

action_logits_t = tf.convert_to_tensor([[0.1, 0.2]])


print(tf.random.categorical(action_logits_t, 1)[0, 0])


print(np.random.choice(2, p=np.squeeze(
    tf.keras.activations.softmax(action_logits_t))))


gamma = 0.99
eps = np.finfo(np.float32).eps.item()


# import gym

# cartPole = gym.make("CartPole-v0")


# value = np.array([[1, 2, 3, 4]])


# print(np.random.choice(2, p=np.squeeze(value)))


# print(tf.random.categorical(value, 1)[0, 0])


# print()
# print(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]).shape)


# x1_input = np.arange(10).reshape(5, 2)

# print(x1_input.shape)

# x1 = tf.keras.layers.Dense(8)(x1_input)


# x2_input = np.arange(10, 20).reshape(5, 2)

# print(x2_input.shape)

# x2 = tf.keras.layers.Dense(8)(x2_input)

# concatted = tf.keras.layers.Concatenate()([x1, x2])
