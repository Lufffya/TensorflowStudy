#
# 文本分类
#

import tensorflow as tf
import numpy as np


imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


print(train_data.shape)
print(train_labels)


# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))


train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=256)


print(len(train_data[0]), len(train_data[1]))


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(10000, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


partial_x_train = train_data[10000:]
partial_y_train = train_labels[10000:]


x_val = train_data[:10000]
y_val = train_labels[:10000]


history = model.fit(partial_x_train, partial_y_train,epochs=40,batch_size=512,validation_data=(x_val, y_val),verbose=1)


results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)