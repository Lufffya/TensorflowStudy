import gym
import numpy as np
import tensorflow as tf

# 定义奖励的折减系数
gamma = 0.99
# 创建CartPole模拟环境
cartPole = gym.make("CartPole-v0")
# 设置环境的随机种子
cartPole.seed(42)
np.random.seed(42)
# 最小的数字是1.0 + 每股收益!=1.0
eps = np.finfo(np.float32).eps.item()

# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# 定义损失函数
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


# 定义动作以及评估模型
def action_critic():
    inputs = tf.keras.layers.Input(shape=(4,))
    common = tf.keras.layers.Dense(128, activation="relu")(inputs)
    action = tf.keras.layers.Dense(2, activation="softmax")(common)
    critic = tf.keras.layers.Dense(1)(common)
    return tf.keras.models.Model(inputs=inputs, outputs=[action, critic])


model = action_critic()
model.summary()


# 运行cartPole环境,模拟推车移动,从环境中获取决策数据
def run_episode(state):
    action_probs, critic_values, rewards = [], [], []
    while True:
        # 可视化过程
        cartPole.render()
        # 将环境状态信息转换为张量作为模型输入
        input_state = np.expand_dims(state, 0)
        # 从环境中预测行动概率和估计未来回报
        action_t, critic_value = model(input_state)
        # 获取动作概率分布中的样本动作
        # action = np.random.categorical(action_logits_t, 1)[0, 0]
        # squeeze 改变数组的维度,这里是从批处理中降维取值
        action = np.random.choice(2, p=np.squeeze(action_t))
        # 将动作应用到当前环境中
        state, reward, done, _ = cartPole.step(action)
        # 记录每一次环境决策的结果
        # action_probs.append(tf.nn.softmax(action_t)[0, action])
        action_probs.append(action_t[0, action])
        critic_values.append(critic_value[0, 0])
        rewards.append(reward)

        if done:
            break
    return action_probs, critic_values, rewards


# 计算每个时间步的预期回报
def get_expected_return(rewards):
    returns = []
    discounted_sum = 0
    for reward in rewards[::-1]:
        discounted_sum = reward + gamma * discounted_sum
        returns.append(discounted_sum)
    # eps是一个很小的非负数
    # 使用eps将可能出现的零用eps来替换，这样不会报错
    # (数组每项 - 所有项均值) / (所有项标准差 + eps)
    return ((returns - np.mean(returns)) / (np.std(returns) + eps))[::-1]


# 定义损失函数计算方法
# 计算 actor-critic 的综合损失值
def compute_loss(action_probs, critic_values, returns):
    actor_loss, critic_loss = [], []
    for _probs, _value, _return in zip(action_probs, critic_values, returns):

        _log_probs = tf.math.log(_probs)

        _diff = _return - _value

        actor_loss.append(-_log_probs * _diff)

        critic_loss.append(huber_loss(tf.expand_dims(
            _value, 0), tf.expand_dims(_return, 0)))

    return sum(actor_loss) + sum(critic_loss)


pisodes = 1
running_reward = 0
while True:
    # 初始化环境
    state = cartPole.reset()
    # 模型训练过程
    with tf.GradientTape() as tape:
        # 运行一个环境片段来收集训练数据
        action_probs, critic_values, rewards = run_episode(state)
        # 计算预期收益
        returns = get_expected_return(rewards)
        # 计算损失函数
        loss = compute_loss(action_probs, critic_values, returns)

    # 根据损失计算梯度值
    grads = tape.gradient(loss, model.trainable_variables)
    # 将梯度值应用于模型的参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = int(tf.math.reduce_sum(rewards))

    running_reward = episode_reward * 0.01 + running_reward * .99

    print(f'episode {pisodes} --- episode_reward: {episode_reward} --- running_reward: {running_reward} --- loss {loss.numpy()}')

    if running_reward > 195:
        break
    pisodes += 1
# while True:
#     action_probs_history = []
#     critic_value_history = []
#     rewards_history = []
#     running_reward = 0
#     episode_count = 0
#     # 初始化环境
#     state = cartPole.reset()
#     # 模型训练过程
#     with tf.GradientTape() as tape:
#         while True:
#             # 可视化过程
#             cartPole.render()
#             # 将环境状态信息转换为张量作为模型输入
#             input_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
#             # 从环境中预测行动概率和估计未来回报
#             action_probs, critic_value = model(input_state)
#             # 获取动作概率分布中的样本动作
#             # ??? 为什么这里要取随机值
#             action = np.random.choice(2, p=np.squeeze(action_probs))
#             # 将动作应用到当前环境中
#             state, reward, done, _ = cartPole.step(action)
#             # 记录每一次决策结果

#             action_probs_history.append(tf.math.log(action_probs[0, action]))
#             critic_value_history.append(critic_value[0, 0])
#             rewards_history.append(reward)
#             if done:
#                 break

#         # Update running reward to check condition for solving
#         running_reward = 0.05 * sum(rewards_history) + \
#             (1 - 0.05) * running_reward

#         # 从奖励中计算期望值
#         # - 在每一个时间步之后，得到的总奖励是多少
#         # - 过去的奖励是用gamma乘以它们而打折的
#         # - 这些是我们评论家的标签
#         returns = []
#         discounted_sum = 0
#         for r in rewards_history[::-1]:
#             discounted_sum = r + gamma * discounted_sum
#             returns.insert(0, discounted_sum)

#         # 规范化
#         returns = np.array(returns)
#         returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
#         returns = returns.tolist()

#         # 计算损失值以更新我们的网络
#         history = zip(action_probs_history, critic_value_history, returns)
#         actor_losses = []
#         critic_losses = []
#         for log_prob, value, ret in history:
#             # 在历史的这一刻，评论家估计我们会得到
#             # 总回报=未来的价值。我们以对数概率采取行动
#             # 最后得到了总奖励。
#             # 必须更新actor，以便它预测导致
#             # 高回报（与评论家的估计相比）具有高概率。
#             diff = ret - value
#             actor_losses.append(-log_prob * diff)  # actor loss

#             # 评论家必须更新，以便能更好地预测
#             # 未来的回报。
#             critic_losses.append(huber_loss(
#                 tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

#         # 反向传播
#         loss_value = sum(actor_losses) + sum(critic_losses)
#         grads = tape.gradient(loss_value, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))

#     episode_count += 1
#     if episode_count % 10 == 0:
#         template = "running reward: {:.2f} at episode {}"
#         print(template.format(running_reward, episode_count))

#     # Condition to consider the task solved
#     if running_reward > 195:
#         print("Solved at episode {}!".format(episode_count))
#         break
