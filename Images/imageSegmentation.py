#
# 图像分割
# https://tensorflow.google.cn/tutorials/images/segmentation


# 图像分割的任务是训练一个神经网络来输出该图像对每一个像素的掩码

from keras.layers import *
from keras.models import *
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 下载 Oxford-IIIT Pets 数据集
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    # 执行 tensorflow 中张量数据类型转换并标准化0-1
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # 像素点在图像分割掩码中被标记为 {1, 2, 3} 中的一个。
    # 将分割掩码都减 1,得到了以下的标签：{0, 1, 2}
    input_mask -= 1
    return input_image, input_mask


def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


# 数据集已经包含了所需的测试集和训练集划分，所以我们也延续使用相同的划分。
TRAIN_LENGTH = info.splits['train'].num_examples  # 训练数据长度
BATCH_SIZE = 64  # 训练数据批次长度
BUFFER_SIZE = 1000  # 缓冲区大小
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE  # 每次批处理的步数

train = dataset['train'].map(
    load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


# 我们来看一下数据集中的一例图像以及它所对应的掩码。
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])


'''定义模型'''

# 每个像素有三种可能的标签
OUTPUT_CHANNELS = 3

'''编码'''
# 编码器是一个预训练的 MobileNetV2 模型
base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False)
base_model.summary()

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.models.Model(
    inputs=base_model.input, outputs=layers, trainable=False)
down_stack.summary()

'''解码'''
# 解码器/升频取样器是简单的一系列升频取样模块，在 TensorFlow examples 中曾被实施过。

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


# u-net是卷积网络体系结构，用于快速精确地分割图像。
# U-Net 由一个编码器（下采样器（downsampler））和一个解码器（上采样器（upsampler））组成
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128

    x = last(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


'''训练模型'''
model = unet_model(OUTPUT_CHANNELS)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])
model.summary()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# 我们试着运行一下模型，看看它在训练之前给出的预测值。
show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS, validation_data=test_dataset, callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


show_predictions(test_dataset, 3)
