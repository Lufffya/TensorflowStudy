import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(6, 1, 1)
y = np.array([item * 2 for item in x], dtype=np.float32).reshape(6, 1, 1)

model = tf.keras.models.Sequential([tf.keras.layers.LSTM(32), tf.keras.layers.Dense(1)])
model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
model.fit(x, y, epochs=1000)