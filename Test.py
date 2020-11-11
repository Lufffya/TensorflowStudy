from tensorflow.keras import layers
from tensorflow import keras
import gym
import numpy as np
import tensorflow as tf

action_logits_t = tf.convert_to_tensor([[0.1,0.2]])


print(tf.random.categorical(action_logits_t, 1)[0, 0])


print(np.random.choice(2, p=np.squeeze(tf.keras.activations.softmax(action_logits_t))))


gamma = 0.99
eps = np.finfo(np.float32).eps.item()


# 计算每个时间步的预期回报
def dddd(rewards):
    returns = []
    discounted_sum = 0
    for reward in rewards[::-1]:
        discounted_sum = reward + gamma * discounted_sum
        returns.append(discounted_sum)

    return ((returns - np.mean(returns)) / (np.std(returns) + eps))[::-1]



# 计算每个时间步的预期回报
def get_expected_return(rewards: tf.Tensor, standardize: bool = True) -> tf.Tensor:

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns



print(get_expected_return([1,1,1]))
print(dddd([1,1,1]))


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

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
# Smallest number such that 1.0 + eps != 1.0
eps = np.finfo(np.float32).eps.item()
num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)
# model = keras.Model(inputs=inputs, outputs=[action, critic])

# 定义动作以及评估模型


def action_critic():
    inputs = tf.keras.layers.Input(shape=(4,))
    common = tf.keras.layers.Dense(128, activation="relu")(inputs)
    action = tf.keras.layers.Dense(2,activation="softmax")(common)
    critic = tf.keras.layers.Dense(1)(common)
    return tf.keras.models.Model(inputs=inputs, outputs=[action, critic])


model = action_critic()

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(tf.convert_to_tensor(15.4545489), model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
