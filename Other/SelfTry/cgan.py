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
num_classes = 10

# mnist手写数据集
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

plt.figure()
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(train_images[i], cmap="gray")
    plt.axis(False)
plt.show()

# 将训练图片的像素压缩至-1到1之间
train_images = (train_images / 127.5) - 1
train_images = train_images.reshape(60000, 784)
train_labels = np.expand_dims(train_labels, axis=-1)

adam = tf.keras.optimizers.Adam(lr=0.0002)

def generator():
    conditional_ipunt = tf.keras.layers.Input(shape=(1,))
    noise_input = tf.keras.layers.Input(shape=(random_dim,))
    c_input = tf.keras.layers.Embedding(num_classes, random_dim)(conditional_ipunt)
    c_input = tf.keras.layers.Flatten()(c_input)
    x = tf.keras.layers.Multiply()([c_input, noise_input])
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    _output = tf.keras.layers.Dense(28 * 28, activation="tanh")(x)
    return tf.keras.models.Model(inputs=[conditional_ipunt, noise_input], outputs=[_output])


def discriminator():
    conditional_ipunt = tf.keras.layers.Input(shape=(1,))
    image_input = tf.keras.layers.Input(shape=(28 * 28,))
    c_input = tf.keras.layers.Embedding(num_classes, 28 * 28)(conditional_ipunt)
    c_input = tf.keras.layers.Flatten()(c_input)
    x = tf.keras.layers.Multiply()([c_input, image_input])
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=[conditional_ipunt, image_input], outputs=[_output])
    model.compile(optimizer=adam, loss="binary_crossentropy")
    return model


# 定义对抗模型,目的是去训练生成模型
def gan(g_model, d_model):
    d_model.trainable = False
    conditional_ipunt = tf.keras.layers.Input(shape=(1,))
    noise_input = tf.keras.layers.Input(shape=(random_dim,))
    noise_image_input = g_model([conditional_ipunt, noise_input])
    _output = d_model([conditional_ipunt, noise_image_input])
    model = tf.keras.models.Model(inputs=[conditional_ipunt, noise_input], outputs=_output)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    return model


generator_model = generator()
discriminator_model = discriminator()
gan_model = gan(generator_model, discriminator_model)


def get_batch_real_samples(batch_size):
    sample_ids = np.random.randint(0, 60000, size=batch_size)
    # 从训练集中随机获取真实的图片
    images_real =  train_images[sample_ids]
    labels_real =  train_labels[sample_ids]
    # 生成真实的标签
    y_real = np.expand_dims(np.ones(batch_size), axis=-1)
    return labels_real, images_real, y_real


def get_batch_fake_samples(batch_size):
    labels_fake = np.expand_dims(np.random.randint(0, num_classes, size=batch_size), axis=-1)
    # 随机生成正态分布的噪点
    noise = np.random.normal(size=[batch_size, random_dim])
    # 生成器生成假的图片
    images_fake = generator_model.predict([labels_fake, noise])
    # 生成假的标签
    y_fake = np.expand_dims(np.zeros(batch_size), axis=-1)
    return labels_fake, images_fake, y_fake


def predict(name):
    labels, noise = [], []
    for i in range(10):
        labels.extend([i] * 10)
        noise.extend(np.random.normal(size=[10, random_dim]))
    
    labels = np.expand_dims(labels, axis=-1)
    noise = np.array(noise)
    generator_Image = generator_model.predict([labels, noise]).reshape(100, 28, 28)
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
        labels_real, images_real, y_real = get_batch_real_samples(int(batch_size / 2))
        labels_fake, images_fake, y_fake = get_batch_fake_samples(int(batch_size / 2))
        
        x1, x2, y =  np.vstack((labels_real, labels_fake)), np.vstack((images_real, images_fake)), np.vstack((y_real, y_fake))

        d_loss = discriminator_model.train_on_batch([x1, x2], y)

        # 训练生成模型
        labels = np.expand_dims(np.random.randint(0, num_classes, size=batch_size), axis=-1)
        noise = np.random.normal(size=[batch_size, random_dim])
        y = np.expand_dims(np.ones(batch_size), axis=-1)
        
        g_loss = gan_model.train_on_batch([labels, noise], y)

        print("epoch:{0} \t g_loss:{1} \t d_loss:{2}".format(epoch, g_loss, d_loss))
        
    predict("epchs{0}.png".format(epoch))


predict()