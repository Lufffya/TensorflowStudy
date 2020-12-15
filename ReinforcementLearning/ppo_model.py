import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models


config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)


class Actor(layers.Layer):
    def __init__(self, action_size, epsilon):
        super(Actor, self).__init__()
        self.epsilon = epsilon
        self.action_size = action_size
        self.dense = layers.Dense(action_size, activation=tf.nn.tanh)
        self.log_std = tf.Variable(tf.zeros(action_size, dtype=tf.float32))

    @tf.function
    def call(self, inputs):
        mu = self.dense(inputs)
        log_std = self.log_std
        return mu, log_std

    @tf.function
    def loss(self, inputs, advantages, actions, logp_old):
        mu, log_std = self(inputs)
        normal = tfp.distributions.Normal(mu, tf.exp(log_std))
        logp = tf.reduce_sum(normal.log_prob(actions), axis=-1, keepdims=True)
        ratio = tf.exp(logp - logp_old)

        pi_loss = tf.reduce_mean(tf.minimum(ratio * advantages, tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages))
        approx_kl = .5 * tf.reduce_mean(tf.square(logp - logp_old))
        entropy_loss = tf.reduce_mean(tf.reduce_sum(normal.entropy(), axis=-1))
        return pi_loss, entropy_loss, tf.reduce_mean(logp_old), tf.reduce_mean(logp), approx_kl, ratio


class Critic(layers.Layer):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense = layers.Dense(1, activation=None)

    @tf.function
    def call(self, inputs):
        value = self.dense(inputs)
        return value

    @tf.function
    def loss(self, inputs, returns):
        value = self(inputs)
        loss = 0.5 * tf.reduce_mean((returns - value) ** 2)
        return loss


class CNN(layers.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = layers.Conv2D(16, 8, 4, padding="valid", activation=tf.nn.leaky_relu)
        self.conv_2 = layers.Conv2D(32, 3, 2, padding="valid", activation=tf.nn.leaky_relu)
        self.flatten = layers.Flatten()

    @tf.function
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.flatten(x)
        return x


class MLP(layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_1 = layers.Dense(64)
        self.dense_2 = layers.Dense(128, activation='relu')

    @tf.function
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x


class PPO(tf.keras.Model):
    def __init__(self, action_size, epsilon, entropy_reg, value_coeff, initial_layer, learning_rate, max_grad_norm):
        super(PPO, self).__init__()
        self.actor = Actor(action_size, epsilon)
        self.critic = Critic()
        self.value_coeff = value_coeff
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        if initial_layer == "MLP":
            self.initial_layer = MLP()
        else:
            self.initial_layer = CNN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        x = self.initial_layer(inputs)
        mu, log_std = self.actor(x)
        value = self.critic(x)
        normal = tfp.distributions.Normal(mu, tf.exp(log_std))
        pi = tf.squeeze(normal.sample(1), axis=0)
        # TODO maybe apply action clipping
        logp_pi = tf.reduce_sum(normal.log_prob(pi), axis=-1, keepdims=True)
        return pi, logp_pi, value

    def loss(self, states, actions, returns, advantages, logp_old):
        with tf.GradientTape() as tape:
            x = self.initial_layer(states)
            value_loss = self.critic.loss(x, returns)
            pi_loss, entropy_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = self.actor.loss(x, advantages, actions, logp_old)
            loss = - pi_loss - entropy_loss * self.entropy_reg + value_loss * self.value_coeff
        grads = tape.gradient(loss, self.trainable_weights)
        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return pi_loss, value_loss, entropy_loss, loss,  old_neg_log_val, neg_log_val, approx_kl, ratio
