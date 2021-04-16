#
# 图像分割
# 图像分割的任务是训练一个神经网络来输出该图像对每一个像素的掩码
# https://tensorflow.google.cn/tutorials/images/segmentation

import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# 什么是图像分割？
# 想知道一个物体在一张图像中的位置、这个物体的形状、以及哪个像素属于哪个物体等等。这种情况下你会希望分割图像

'''准备数据'''
# 下载 Oxford-IIIT Pets 数据集
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, download=False)


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


# 数据集已经包含了所需的测试集和训练集划分
train_length = info.splits['train'].num_examples  # 训练数据长度
batch_size = 64  # 训练数据批次长度
steps_per_epoch = train_length // batch_size  # 每次批处理的步数

# 对数据进行处理
train_dataset = dataset['train'].map(load_image_train).cache().shuffle(1000).batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = dataset['test'].map(load_image_test).batch(batch_size)


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# 打印X(图片)Y(图片对应的像素类别掩码)
train = dataset['train'].map(load_image_train)
for image, mask in train.take(3):
    sample_image, sample_mask = image, mask
    display([sample_image, sample_mask])


'''定义模型'''

# 图片像素标签的类别
output_channels = 3
input_shape = (128, 128, 3)


#################编码器模型（下采样）###################
# 编码器是一个预训练的 MobileNetV2 模型
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
# base_model.summary()

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project'       # 4x4
]

# 获取网络层
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=layers, trainable=False)
# down_stack.summary()


#################解码器模型（上采样）###################
# 解码器/升频取样器是简单的一系列升频取样模块
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3)    # 32x32 -> 64x64
]


#################构建模型###################
# u-net是卷积网络体系结构，用于快速精确地分割图像。
# U-Net 由一个编码器（下采样器（downsampler））和一个解码器（上采样器（upsampler））组成
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=input_shape)
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
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128

    x = last(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


'''训练模型'''
model = unet_model(output_channels)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


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
        display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# 我们试着运行一下模型，看看它在训练之前给出的预测值。
show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


val_subsplits = 5
validation_steps = info.splits['test'].num_examples // batch_size // val_subsplits

model_history = model.fit(train_dataset, epochs=10, validation_steps=validation_steps, steps_per_epoch=steps_per_epoch, validation_data=test_dataset, callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(10)
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
