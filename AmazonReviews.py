
import numpy as np
import tensorflow as tf
import pandas as pd



reviews = pd.read_csv("DataSet\AmazonReviews(B07PBV7D48).csv")


init_checkpoint='Models\\albert_base_v2\\albert_base\\model.ckpt-best'
model=tf.keras.Model() # Bert pre-trained model as feature extractor.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(init_checkpoint)


model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.SparseCategoricalCrossentropy(),metrics=tf.metrics.Accuracy())


model.fit(reviews,epochs=10)



print(reviews.info())


