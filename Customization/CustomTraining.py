#
# 自定义的训练
# 这个教程将利用机器学习的手段来对鸢尾花按照物种进行分类
# https://tensorflow.google.cn/tutorials/customization/custom_training_walkthrough

import os
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 本教程将利用 TensorFlow 来进行以下操作：
# 构建一个模型
# 用样例数据集对模型进行训练，以及
# 利用该模型对未知数据进行预测

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# 鸢尾花分类问题

# train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
# train_dataset_fp = tf.keras.utils.get_file(
#     fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
# print("Local copy of the dataset file: {}".format(train_dataset_fp))

# CSV文件中列的顺序
column_names = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

# 0 ：山多
# 1 : 变色鸢尾
# 2 : 维吉尼亚鸢尾
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# 由于数据集是 CSV 格式的文本文件，请使用 make_csv_dataset 函数将数据解析为合适的格式
batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(
    "DataSet\iris_training.csv", batch_size, column_names=column_names, label_name=label_name, num_epochs=1)

# make_csv_dataset 返回一个(features, label) 对构建的 tf.data.Dataset ，
# 其中 features 是一个字典: {'feature_name': value}
features, labels = next(iter(train_dataset))
print(features)
print(labels)

# 注意到具有相似特征的样本会归为一组，即分为一批。更改 batch_size 可以设置存储在这些特征数组中的样本数。
# 绘制该批次中的几个特征后，就会开始看到一些集群现象：
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

# 要简化模型构建步骤，请创建一个函数以将特征字典重新打包为形状为 (batch_size, num_features) 的单个数组。
# 此函数使用 tf.stack 方法，该方法从张量列表中获取值，并创建指定维度的组合张量:


def pack_features_vector(features, labels):
    """将特征打包到一个数组中"""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


# 然后使用 tf.data.Dataset.map 方法将每个 (features,label) 对中的 features 打包到训练数据集中：
train_dataset = train_dataset.map(pack_features_vector)

# Dataset 的特征元素被构成了形如 (batch_size, num_features) 的数组。我们来看看前几个样本:
# 32个特征和32个标签
features, labels = next(iter(train_dataset))

# print(features[:5])
# print(labels[:5])

# 使用 Keras 创建模型

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu,
                          input_shape=(4,)),  # 需要给出输入的形式
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])


# 使用模型
predictions = model(features)
print(predictions[:5])

# 在此示例中，每个样本针对每个类别返回一个 logit。
# 要将这些对数转换为每个类别的概率，请使用 softmax 函数:

tf.nn.softmax(predictions[:5])
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


'''训练模型'''


# 定义损失函数
def loss(model, x, y):
    # 将真实的X丢到模型,得到预测的Y
    y_ = model(x)
    # 定义损失函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    # 拿到真实Y和预测的Y计算出损失函数
    return loss_object(y_true=y, y_pred=y_)


# 输出损失函数
# l = loss(model, features, labels)
# print("Loss test: {}".format(l))


# 使用 tf.GradientTape 的前后关系来计算梯度以优化你的模型:
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        # 计算损失函数
        loss_value = loss(model, inputs, targets)
        # 根据损失函数和模型训练参数计算梯度
        gradient = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradient


# 定义优化器 learning_rate 优化速率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# 我们将使用它来计算单个优化步骤：


# 得到损失函数和梯度
#loss_value, grads = grad(model, features, labels)

# print("Step: {}, Initial Loss: {}".format(
# optimizer.iterations.numpy(), loss_value.numpy()))

# 将计算出的梯度应用到优化器
# optimizer.apply_gradients(zip(grads, model.trainable_variables))

# print("Step: {},         Loss: {}".format(
#     optimizer.iterations.numpy(), loss(model, features, labels).numpy()))


# 使用相同的模型变量重新运行此单元

# 保留结果用于绘制
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # 训练循环-使用32个批次
    for x, y in train_dataset:
        '''优化模型'''
        # 计算出损失函数和梯度值
        loss_value, grads = grad(model, x, y)
        # 将梯度应用到优化器,自动求导,调整参数向前传播
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 追踪进度
        epoch_loss_avg(loss_value)  # 添加当前的 batch loss
        # 比较预测标签与真实标签
        epoch_accuracy(y, model(x))

    # 循环结束
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))


# 可视化损失函数随时间推移而变化的情况
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


# 评估模型的效果
# 建立测试数据集
test_dataset = tf.data.experimental.make_csv_dataset(
    "DataSet\iris_test.csv", batch_size, column_names=column_names, label_name='species', num_epochs=1, shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)


# 根据测试数据集评估模型
# 与训练阶段不同，模型仅评估测试数据的一个周期。
# 在以下代码单元格中，我们会遍历测试集中的每个样本，然后将模型的预测结果与实际标签进行比较。这是为了衡量模型在整个测试集中的准确率
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
print(tf.stack([y, prediction], axis=1))


# 使用经过训练的模型进行预测
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
