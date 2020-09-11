
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import scipy.spatial
from tqdm import tqdm
import bert 

# Your choice here.
max_seq_length = 128

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_word_ids")

input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_mask")

segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="segment_ids")

bert_layer = hub.KerasLayer("Models\Bert_EN_Uncased_L-12_H-768_A-12_2_TF-Hubs",trainable=True)

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file,do_lower_case)


# 读取数据集
train_data = pd.read_csv("DataSet\AmazonReviews(B07PBV7D48).csv")

train_X = []

for review in train_data.values:
    tokens = tokenizer.tokenize(list(review))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) < max_seq_length:
        token_ids = token_ids + [tokenizer.vocab["[unused1]"]] * (max_seq_length - len(token_ids))
    token_ids = token_ids[:max_seq_length]
    train_X.append(np.array(token_ids))

train_X = np.array(train_X)



def Get_Encoding(inputs):
    outPut = []
    pooling = tf.keras.layers.GlobalAveragePooling1D()
    for i in tqdm(range(0,len(inputs),32)):
        batch = inputs[i:i+32]
        batch_word_embedding = pooling(bert_layer(batch))
        outPut.extend(batch_word_embedding.numpy())
    return np.array(outPut)

# 解析标签
target = ["味道好，送餐速度也快"]
target_Token = tokenizer.tokenize(target[0])
target_ids = tokenizer.convert_tokens_to_ids(target_Token)
if len(target_ids) < max_seq_length:
    target_ids = target_ids + [tokenizer.vocab["[unused1]"]] * (max_seq_length - len(target_ids))
    target_ids = target_ids[:max_seq_length]
target_ids = np.array([np.array(target_ids)])

# 获取标签句子的词向量
target_X_Encoding = Get_Encoding(target_ids)

# 获所有训练句子的词向量
train_X_Encoding = Get_Encoding(train_X)

# 计算标签句子词向量和训练句子词向量之间的欧式距离
euclidean_Distance = scipy.spatial.distance.cdist(target_X_Encoding, train_X_Encoding)[0]

_zip = zip(range(len(euclidean_Distance)),euclidean_Distance)

_sorted = sorted(_zip,key=lambda x: x[1],reverse=False)

for _index, distance in _sorted[0:20]:
    print(train_data.review.values[_index], "(Distance: %.4f)" % (distance))

print()






# init_checkpoint='Models\\albert_base_v2\\albert_base\\model.ckpt-best'
# model=tf.keras.Model() # Bert pre-trained model as feature extractor.
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(init_checkpoint)

# model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.SparseCategoricalCrossentropy(),metrics=tf.metrics.Accuracy())

# model.fit(reviews,epochs=10)

print(reviews.info())


