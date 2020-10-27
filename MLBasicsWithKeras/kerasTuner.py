#
# 使用 Keras Tuner 调整超参数
# https://tensorflow.google.cn/tutorials/keras/keras_tuner
# https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html


# Keras Tuner是一个库，可帮助您为TensorFlow程序选择最佳的超参数集。选择合适的一组超参数，为您的机器学习（ML）应用程序的过程被称为超参数调整或hypertuning。

# 超参数是控制训练过程和ML模型拓扑的变量。这些变量在训练过程中保持不变，并直接影响ML程序的性能。超参数有两种类型：

# 影响模型选择的模型超参数，例如隐藏层的数量和宽度
# 影响学习算法的速度和质量的算法超参数，例如随机梯度下降（SGD）的学习率和ak最近邻（KNN）分类器的最近邻数
# 在本教程中，您将使用Keras Tuner对图像分类应用程序执行超调。

import tensorflow as tf
from tensorflow import keras

import IPython
import kerastuner as kt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 在本教程中，您将使用Keras Tuner为机器学习模型找到最佳的超参数，该模型对Fashion MNIST数据集中的服装图像进行分类。
(img_train, label_train), (img_test,
                           label_test) = keras.datasets.fashion_mnist.load_data()
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

# 定义模型
# 在本教程中，您将使用模型构建器功能来定义图像分类模型。模型构建器函数返回已编译的模型，并使用您内联定义的超参数对模型进行超调。


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # 调整第一个密集层中的单位数
    # 在32-512之间选择一个最佳值
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # 调整优化器的学习速率
    # 从0.01、0.001或0.0001中选择一个最佳值
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

# 实例化调谐器以执行超调谐。该Keras调谐器有四个可用的调谐器-
# RandomSearch，Hyperband，BayesianOptimization和Sklearn。在本教程中，您将使用Hyperband调谐器


# 要实例化Hyperband调谐器，您必须指定超模型，objective要优化的以及要训练的最大时期数（max_epochs）
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')


# 在运行超参数搜索之前，定义一个回调以在每个训练步骤结束时清除训练输出。
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


# 运行超参数搜索。tf.keras.model.fit除了上面的回调外，search方法的参数与所使用的参数相同。
tuner.search(img_train, label_train, epochs=10, validation_data=(
    img_test, label_test), callbacks=[ClearTrainingOutput()])

# 得到最优超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
超参数搜索已完成。第一密连通的最优单元数
层是{best_hps.get('units')}，优化程序的最佳学习速率
是{best_hps.get('learning_rate')}。
""")

# 用最优超参数建立模型，并对其进行训练
model = tuner.hypermodel.build(best_hps)

model.fit(img_train, label_train, epochs=10,
          validation_data=(img_test, label_test))
model.evaluate(img_test, label_test)
