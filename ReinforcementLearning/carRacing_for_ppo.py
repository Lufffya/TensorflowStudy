import cv2
import numpy as np
import tensorflow as tf
from scipy.signal import lfilter
from ppo_model import PPO
import gym
from collections import deque


config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

#==========定义超参数============#
GAMMA = 0.99
LAMBDA = 0.95


class CarRacing():
    def __init__(self):
        self.env = gym.make("CarRacing-v0")
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = 4
        self.stacked_frames = deque(maxlen=self.img_stack)

    def preprocess_frame(self, frame):
        frame = frame[0:84, :, :]
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.asarray(frame, np.float32) / 255
        frame = np.reshape(frame, (64, 64, 1))
        return frame

    def reset(self):
        state = self.env.reset()
        frame = self.preprocess_frame(state)
        for _ in range(self.img_stack):
            self.stacked_frames.append(frame)
        return np.concatenate(self.stacked_frames, axis=2)

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        frame = self.preprocess_frame(state)
        self.stacked_frames.append(frame)
        return np.concatenate(self.stacked_frames, axis=2), reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def action_space_shape(self):
        return self.env.action_space.shape[0]

    def action_space_low(self):
        return self.env.action_space.low

    def action_space_high(self):
        return self.env.action_space.high


class Agent():

    def __init__(self, env):
        self.env = env
        self.buffer = []
        self.EPOCHS = 4
        self.STEPS_PER_BATCH = 128
        self.STEPS_PER_EPOCH = 512
        self.ACTION_SIZE = self.env.action_space_shape()
        self.EPSILON = 0.2
        self.ENTROPY_REG = 0.01
        self.VALUE_COEFFICIENT = 0.5
        self.LEARNING_RATE = 3e-4
        self.MAX_GRAD_NORM = 0.5
        self.model = PPO(self.ACTION_SIZE, self.EPSILON, self.ENTROPY_REG,
                         self.VALUE_COEFFICIENT, "CNN", self.LEARNING_RATE, self.MAX_GRAD_NORM)

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        pi, old_log_p, value = self.model.call(state)
        return pi, old_log_p, value

    def store(self, state, action, value, reward, done, old_log_pi):
        self.buffer.append((state, action, value, reward, done, old_log_pi))
        if len(self.buffer) == self.STEPS_PER_EPOCH:
            return True
        else:
            return False

    def get_store(self):
        states, actions, values, rewards, dones, old_log_pi = [], [], [], [], [], []
        for state, action, value, reward, done, _old_log_pi in self.buffer:
            states.append(state)
            actions.append(action)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            old_log_pi.append(_old_log_pi)

        return np.array(states), np.array(actions), np.array(values), np.array(rewards), np.array(dones), np.array(old_log_pi)

    def compute_gae(self, rewards, values, bootstrap_values, dones, gamma, lam):
        values = np.vstack((values, [bootstrap_values]))
        deltas = []
        for i in reversed(range(len(rewards))):
            V = rewards[i] + (1.0 - dones[i]) * gamma * values[i + 1]
            delta = V - values[i]
            deltas.append(delta)
        deltas = np.array(list(reversed(deltas)))

        A = deltas[-1, :]
        advantages = [A]
        for i in reversed(range(len(deltas) - 1)):
            A = deltas[i] + (1.0 - dones[i]) * gamma * lam * A
            advantages.append(A)
        advantages = reversed(advantages)
        advantages = np.array(list(advantages))
        return advantages

    def update(self, last_val):
        states, actions, values, rewards, dones, old_log_prob = self.get_store()

        advantages = self.compute_gae(rewards, values, last_val, dones, GAMMA, LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        indexs = np.arange(self.STEPS_PER_EPOCH)
        for epoch in range(self.EPOCHS):
            np.random.shuffle(indexs)
            for start in range(0, self.STEPS_PER_EPOCH, self.STEPS_PER_BATCH):
                end = start + self.STEPS_PER_BATCH
                batch_indexs = indexs[start:end]
                slices = (arr[batch_indexs] for arr in (states, actions, returns, advantages, old_log_prob))
                pi_loss, value_loss, entropy_loss, total_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = self.model.loss(*slices)

        self.buffer = []

    
def main():
    # 初始化环境
    env = CarRacing()
    # 初始化智能体
    agent = Agent(env)

    i_episode = 0
    # 游戏最长集数
    while True:
        i_episode += 1
        # 总得分
        total_reward = 0
        # 重置游戏环境
        state = env.reset()
        # 时间步长
        t = 0
        while True:
            t += 1
            # 可视化环境
            env.render()

            pi, old_log_pi, value = agent.select_action(state)

            action = np.clip(pi.numpy()[0], env.action_space_low(), env.action_space_high())

            next_state, reward, done = env.step(action)

            if agent.store(state, pi.numpy()[0], value.numpy()[0], reward, done, old_log_pi.numpy()[0]):
                pi, old_log_pi, value = agent.select_action(next_state)
                last_value =  value.numpy()[0]
                agent.update(last_value)

            state = next_state
            total_reward += reward

            if done:
                break

    env.close()


if __name__ == "__main__":
    main()
