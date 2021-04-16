import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

# 随机维度
random_Dim = 100
# 训练数据批次
batch_Size = 60000
# 训练次数
epochs = 2000

# mnist手写数据集
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()


def SeeTrain_images():
    plt.figure()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(train_images[i], cmap="gray")
        plt.axis(False)
    plt.show()


SeeTrain_images()

# 共用相同的优化器
optimizer = tf.keras.optimizers.Adam(lr=0.0003)

# 定义生成模型
generator_Model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(784, activation="tanh"),
])
generator_Model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003), loss="binary_crossentropy")


# 定义判别模型
discriminator_Model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(128,),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
discriminator_Model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003), loss="binary_crossentropy")


# 定义对抗模型,目的是去训练生成模型
discriminator_Model.trainable = False
gan_Input = tf.keras.layers.Input(shape=(random_Dim,))
x = generator_Model(gan_Input)
gan_Output = discriminator_Model(x)
gan_Model = tf.keras.models.Model(inputs=gan_Input, outputs=gan_Output)
gan_Model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0003))


# 将训练图片的像素压缩至-1到1之间
train_images = (train_images / 127.5) - 1
train_images = train_images.reshape(60000, 784)


def Get_Batch_Real_Samples():
    # 从训练集中随机获取真实的图片
    x_Real = train_images[np.random.randint(0, 60000, size=batch_Size)]
    # 生成真实的标签
    y_Real = np.ones(batch_Size)
    return x_Real, y_Real


def Get_Batch_Fake_Samples():
    # 随机生成正态分布的噪点
    noise = np.random.normal(-1, 1, size=[batch_Size, random_Dim])
    # 生成器生成假的图片
    x_Fake = generator_Model(noise)
    # 生成假的标签
    y_Fake = np.zeros(batch_Size)
    return x_Fake, y_Fake


def Predict(name="20000.png"):
    generator_Input = np.random.normal(-1, 1, size=[60000, random_Dim])
    generator_Predict = generator_Model(generator_Input)
    generator_Image = np.array(generator_Predict).reshape(60000, 28, 28)
    plt.figure()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(generator_Image[i], cmap="gray")
        plt.axis(False)
    plt.savefig(name)
    # plt.show()


for epoch in range(1, epochs+1):

    for step in range(int(60000 / batch_Size)):

        '''训练辨别模型'''
        x_Real, y_Real = Get_Batch_Real_Samples()

        d_loss1 = discriminator_Model.train_on_batch(x_Real, y_Real)

        x_Fake, y_Fake = Get_Batch_Fake_Samples()

        d_loss2 = discriminator_Model.train_on_batch(x_Fake, y_Fake)

        d_loss = d_loss1 + d_loss2

        '''训练生成模型'''
        noise = np.random.normal(-1, 1, size=[batch_Size, random_Dim])

        g_loss = gan_Model.train_on_batch(noise, np.ones(batch_Size))

        if epoch == 1 or epoch % 50 == 0:
            Predict("epchs{0}.png".format(epoch))

    print("epoch:{0},g_loss:{1},d_loss:{2}".format(epoch, g_loss, d_loss))


Predict()
plt.show()
