#
# 保存和加载模型
# https://tensorflow.google.cn/tutorials/keras/save_and_load

import os
import tensorflow as tf
from tensorflow import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(tf.version.VERSION)


# 获取示例数据集
# 要演示如何保存和加载权重，您将使用 MNIST 数据集. 要加快运行速度，请使用前1000个示例：
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 定义一个简单的序列模型
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    return model


# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()


# 在训练期间保存模型（以 checkpoints 形式保存）
# 您可以使用训练好的模型而无需从头开始重新训练，或在您打断的地方开始训练，以防止训练过程没有保存。 tf.keras.callbacks.ModelCheckpoint 允许在训练的过程中和结束时回调保存的模型。
# Checkpoint 回调用法
# 创建一个只在训练期间保存权重的 tf.keras.callbacks.ModelCheckpoint 回调：

checkpoint_path = "MLBasicsWithKeras/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1)

# 使用新的回调训练模型
model.fit(train_images, train_labels, epochs=10, validation_data=(
    test_images, test_labels), callbacks=[cp_callback])  # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）
# 是防止过时使用，可以忽略。

# 创建一个新的未经训练的模型。仅恢复模型的权重时，必须具有与原始模型具有相同网络结构的模型。由于模型具有相同的结构，您可以共享权重，尽管它是模型的不同实例。

# 现在重建一个新的未经训练的模型，并在测试集上进行评估。未经训练的模型将在机会水平（chance levels）上执行（准确度约为10％）：
# 创建一个基本模型实例
model = create_model()

# 评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 然后从 checkpoint 加载权重并重新评估：
# 加载权重
model.load_weights(checkpoint_path)

# 重新评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# checkpoint 回调选项
# 回调提供了几个选项，为 checkpoint 提供唯一名称并调整 checkpoint 频率。

# 训练一个新模型，每五个 epochs 保存一次唯一命名的 checkpoint ：
# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "MLBasicsWithKeras/training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# 创建一个新的模型实例
model = create_model()

# 使用 `checkpoint_path` 格式保存权重
# model.save_weights(checkpoint_path.format(epoch=0))

# 使用新的回调训练模型
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)


# 注意: 默认的 tensorflow 格式仅保存最近的5个 checkpoint 。
# 如果要进行测试，请重置模型并加载最新的 checkpoint ：
# 创建一个新的模型实例
model = create_model()

# 加载以前保存的权重
model.load_weights(latest)

# 重新评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# 手动保存权重
# 保存权重
# model.save_weights('./checkpoints/my_checkpoint')

# 创建模型实例
# model = create_model()

# 恢复权重
# model.load_weights('./checkpoints/my_checkpoint')

# 评估模型
# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# 保存整个模型

# SavedModel 格式
# SavedModel 格式是序列化模型的另一种方法。以这种格式保存的模型，可以使用 tf.keras.models.load_model 还原，并且模型与 TensorFlow Serving 兼容。
# SavedModel 指南详细介绍了如何提供/检查 SavedModel。以下部分说明了保存和还原模型的步骤。
# 创建并训练一个新的模型实例。
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# 将整个模型另存为 SavedModel。
model.save('MLBasicsWithKeras/saved_model')

new_model = tf.keras.models.load_model('MLBasicsWithKeras/saved_model')
# 检查其架构
new_model.summary()
# 还原的模型使用与原始模型相同的参数进行编译。 尝试使用加载的模型运行评估和预测：
# 评估还原的模型
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
print(new_model.predict(test_images).shape)


# HDF5 格式
# Keras使用 HDF5 标准提供了一种基本的保存格式。
# 创建并训练一个新的模型实例
model = create_model()
model.fit(train_images, train_labels, epochs=5)
# 将整个模型保存为 HDF5 文件。
# '.h5' 扩展名指示应将模型保存到 HDF5。
model.save('MLBasicsWithKeras/my_model.h5')

# 现在，从该文件重新创建模型：
# 重新创建完全相同的模型，包括其权重和优化程序
new_model = tf.keras.models.load_model('MLBasicsWithKeras/my_model.h5')
# 显示网络结构
new_model.summary()
# 检查其准确率（accuracy）：
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
