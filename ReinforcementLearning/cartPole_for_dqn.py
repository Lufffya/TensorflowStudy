#
# DQN(Deep Q-learning Network) 实现 CartPole
#

from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',optimizer=Adam(lr=0.001))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []

        x_true, x_pred = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])

            x_true.append(target_f[0])
            x_pred.append(self.model.predict(state)[0])

        loss = tf.keras.losses.MSE(np.array(x_pred),np.array(x_true))

        # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)

        # Keeping track of loss
        # loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    batch_size = 32

    i_episode = 0
    while True:
        i_episode += 1
        step = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        loss = 0
        isloss = False
        with tf.GradientTape() as tape:

            while True:
                step += 1
                # env.render()

                action = agent.act(state)

                next_state, reward, done, _ = env.step(action)

                reward = reward if not done else -10

                next_state = np.reshape(next_state, [1, state_size])

                agent.memorize(state, action, reward, next_state, done)

                state = next_state

                if done:
                    break

        if len(agent.memory) > batch_size:
            loss = agent.replay(batch_size)
            isloss=True

        if isloss:

            grads = tape.gradient(loss, agent.model.trainable_variables)

            optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

        print("episode: {}, time: {}, loss: {:.4f}".format(i_episode, step, loss))


# import random
# import gym
# import numpy as np
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from scores.score_logger import ScoreLogger

# ENV_NAME = "CartPole-v1"

# GAMMA = 0.95
# LEARNING_RATE = 0.001

# MEMORY_SIZE = 1000000
# BATCH_SIZE = 20

# EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.01
# EXPLORATION_DECAY = 0.995


# class DQNSolver:

#     def __init__(self, observation_space, action_space):
#         self.exploration_rate = EXPLORATION_MAX

#         self.action_space = action_space
#         self.memory = deque(maxlen=MEMORY_SIZE)

#         self.model = Sequential()
#         self.model.add(Dense(24, input_shape=(
#             observation_space,), activation="relu"))
#         self.model.add(Dense(24, activation="relu"))
#         self.model.add(Dense(self.action_space, activation="linear"))
#         self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() < self.exploration_rate:
#             return random.randrange(self.action_space)
#         q_values = self.model.predict(state)
#         return np.argmax(q_values[0])

#     def experience_replay(self):
#         if len(self.memory) < BATCH_SIZE:
#             return
#         batch = random.sample(self.memory, BATCH_SIZE)
#         for state, action, reward, state_next, terminal in batch:
#             q_update = reward
#             if not terminal:
#                 q_update = (reward + GAMMA *
#                             np.amax(self.model.predict(state_next)[0]))
#             q_values = self.model.predict(state)
#             q_values[0][action] = q_update
#             self.model.fit(state, q_values, verbose=0)
#         self.exploration_rate *= EXPLORATION_DECAY
#         self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


# def cartpole():
#     env = gym.make(ENV_NAME)
#     # score_logger = ScoreLogger(ENV_NAME)
#     observation_space = env.observation_space.shape[0]
#     action_space = env.action_space.n
#     dqn_solver = DQNSolver(observation_space, action_space)
#     run = 0
#     while True:
#         run += 1
#         state = env.reset()
#         state = np.reshape(state, [1, observation_space])
#         step = 0
#         while True:
#             step += 1
#             env.render()
#             action = dqn_solver.act(state)
#             state_next, reward, terminal, info = env.step(action)
#             reward = reward if not terminal else -reward
#             state_next = np.reshape(state_next, [1, observation_space])
#             dqn_solver.remember(state, action, reward, state_next, terminal)
#             state = state_next
#             if terminal:
#                 print("Run: " + str(run) + ", exploration: " +
#                       str(dqn_solver.exploration_rate) + ", score: " + str(step))
#                 # score_logger.add_score(step, run)
#                 break
#             dqn_solver.experience_replay()


# if __name__ == "__main__":
#     cartpole()
