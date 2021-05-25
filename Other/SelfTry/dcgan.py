import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

# 随机维度
random_dim = 100
# 训练数据批次
batch_size = 256
# 训练次数
epochs = 500

# mnist手写数据集
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

plt.figure()
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(train_images[i], cmap="gray")
    plt.axis(False)
plt.show()

# 将训练图片的像素压缩至-1到1之间
train_images = (train_images / 127.5) - 1
train_images = np.expand_dims(train_images, axis=-1)

adam = tf.keras.optimizers.Adam(lr=0.0002)

def generator():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7*7*128, input_shape=(random_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(64, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(1, (3,3), strides=(2,2), padding="same", activation="tanh")
    ])
    return model


def discriminator():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (3,3), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=adam, loss="binary_crossentropy")
    return model


# 定义对抗模型,目的是去训练生成模型
def gan(g_model, d_model):
    d_model.trainable = False
    _input = tf.keras.layers.Input(shape=(random_dim,))
    x = g_model(_input)
    _output = d_model(x)
    model = tf.keras.models.Model(inputs=_input, outputs=_output)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    return model


generator_model = generator()
discriminator_model = discriminator()
gan_model = gan(generator_model, discriminator_model)


def get_batch_real_samples(batch_size):
    # 从训练集中随机获取真实的图片
    x_real =  train_images[np.random.randint(0, 60000, size=batch_size)]
    # 生成真实的标签
    y_real = np.expand_dims(np.ones(batch_size), axis=-1)
    return x_real, y_real


def get_batch_fake_samples(batch_size):
    # 随机生成正态分布的噪点
    noise = np.random.normal(size=[batch_size, random_dim])
    # 生成器生成假的图片
    x_fake = generator_model.predict(noise)
    # 生成假的标签
    y_fake = np.expand_dims(np.zeros(batch_size), axis=-1)
    return x_fake, y_fake


def predict(name):
    generator_Input = np.random.normal(size=[100, random_dim])
    generator_Image = np.squeeze(generator_model.predict(generator_Input))
    plt.figure()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(generator_Image[i], cmap="gray")
        plt.axis(False)
    plt.savefig(name)
    # plt.show()


for epoch in range(epochs):

    for step in range(int(60000 / batch_size)):

        # 训练辨别模型
        x_real, y_real = get_batch_real_samples(int(batch_size / 2))

        x_fake, y_fake = get_batch_fake_samples(int(batch_size / 2))
        
        x, y =  np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))

        d_loss = discriminator_model.train_on_batch(x, y)

        # 训练生成模型
        noise = np.random.normal(size=[batch_size, random_dim])

        g_loss = gan_model.train_on_batch(noise, np.expand_dims(np.ones(batch_size), axis=-1))

        print("epoch:{0} \t g_loss:{1} \t d_loss:{2}".format(epoch, g_loss, d_loss))
        
    predict("epchs{0}.png".format(epoch))


predict()