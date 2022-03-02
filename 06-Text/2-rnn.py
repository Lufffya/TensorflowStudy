#! -*- coding:utf-8 -*-
# 使用 RNN 进行文本分类
# 本文本分类教程在IMDB 大型电影评论数据集上训练一个循环神经网络, 用于情感分析.

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

# 导入matplotlib并创建一个辅助函数来绘制图形:
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


# 设置输入管道

# IMDB 大型电影评论数据集是一个二元分类数据集——所有评论都有正面或负面情绪.

# 使用TFDS下载数据集. 有关如何手动加载此类数据的详细信息, 请参阅加载文本教程.

dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
print(train_dataset.element_spec)