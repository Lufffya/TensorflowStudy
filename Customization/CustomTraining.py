#
# 自定义的训练
# 这个教程将利用机器学习的手段来对鸢尾花按照物种进行分类
# https://tensorflow.google.cn/tutorials/customization/custom_training_walkthrough

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
