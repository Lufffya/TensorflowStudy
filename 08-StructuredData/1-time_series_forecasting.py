#
# https://www.tensorflow.org/tutorials/structured_data/time_series
#


import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


''' 天气数据集 '''
# 该数据集包含 14 个不同的特征, 例如气温,大气压力和湿度
# 从 2003 年开始, 每 10 分钟收集一次这些数据. 为了提高效率，您将仅使用 2009 年至 2016 年期间收集的数据
# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='E:\zhaoxinchen\jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
csv_path = r"E:\zhaoxinchen\jena_climate_2009_2016.csv.zip"

# 本教程将只处理每小时预测, 因此首先从 10 分钟间隔到 1 小时对数据进行二次采样:

df = pd.read_csv(csv_path)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

# 让我们看一下数据. 这是前几行:
print(df.head())

# 以下是一些功能随时间的演变
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)
plt.show()

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
plt.show()


### 检查和清理 ###
# 接下来看数据集的统计:
# wv (m/s) 风速(米/秒)
print(df.describe().transpose())

# 风速
# 应该突出的一件事是min风速wv (m/s)和max. wv (m/s)列的值. 
# 这-9999很可能是错误的. 有一个单独的风向列, 所以速度应该是>=0. 用零替换它

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame
print(df['wv (m/s)'].min())


### 特征工程 ###
# 在深入构建模型之前, 了解您的数据很重要, 并确保您传递的是模型格式正确的数据

# 风
# 数据的最后一列wd (deg)以度为单位给出风向. 角度不是很好的模型输入, 360° 和 0° 应该相互靠近, 并且平滑地环绕. 如果没有风, 方向应该无关紧要
# 现在风数据的分布是这样的:
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
plt.show()

# 但是, 如果将风向和速度列转换为风向量, 模型将更容易解释:
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# 转换为弧度
wd_rad = df.pop('wd (deg)')*np.pi / 180

# 计算风的 x 和 y 分量
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# 计算最大风 x 和 y 分量
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

# 风向量的分布对于模型正确解释要简单得多
plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
print(ax.axis('tight'))
plt.show()


# 时间
# 同样, 该Date Time列非常有用, 但不是这种字符串形式. 首先将其转换为秒:
timestamp_s = date_time.map(pd.Timestamp.timestamp)

# 与风向类似, 以秒为单位的时间不是有用的模型输入. 作为天气数据, 它具有明确的每日和每年的周期性. 有很多方法可以处理周期性
# 一个简单的方法将其转换成可用信号是使用sin和cos以清除 "一天的时间" 和 "一年的时间" 信号的时间转换:
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.show()


# 这使模型可以访问最重要的频率特征. 在这种情况下, 您提前知道哪些频率很重要
# 如果您不知道, 您可以使用fft. 
# 为了检查我们的假设, 这里是tf.signal.rfft温度随时间的变化. 注意在1/year和附近频率处的明显峰值1/day:
fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
plt.show()


### 拆分数据 ###
# 我们将对(70%, 20%, 10%)训练,验证和测试集使用拆分. 请注意, 数据在拆分之前没有被随机打乱. 这是出于两个原因
# 它确保仍然可以将数据切割成连续样本的窗口
# 它确保验证/测试结果更加真实, 并根据模型训练后收集的数据进行评估
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]


### 规范化数据 ###
# 在训练神经网络之前缩放特征很重要. 标准化是进行这种缩放的常用方法. 减去平均值并除以每个特征的标准差
# 均值和标准差应仅使用训练数据计算, 以便模型无法访问验证和测试集中的值
# 同样有争议的是, 模型在训练时不应该访问训练集中的未来值, 并且这种归一化应该使用移动平均线来完成
# 这不是本教程的重点, 验证和测试集可确保您获得（某种程度上）真实的指标. 因此, 为了简单起见, 本教程使用了一个简单的平均值
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# 现在看一下特征的分布. 有些特征确实有长尾, 但没有像-9999风速值这样的明显误差
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.show()


''' 数据窗口 '''
# 本教程中的模型将根据数据中的连续样本窗口进行一组预测
# 输入窗口的主要特点是:
#   输入和标签窗口的宽度（时间步数）
#   它们之间的时间偏移
#   哪些特征用作输入, 标签或两者

# 本教程构建了多种模型（包括线性、DNN、CNN 和 RNN 模型）, 并将它们用于两者:
# 单输出和多输出预测
# 单时间步和多时间步预测

# 本节重点介绍实现数据窗口, 以便它可以重用于所有这些模型
# 根据任务和模型类型, 您可能希望生成各种数据窗口. 这里有些例子:
# https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing

# 本节的其余部分定义了一个WindowGenerator类. 这个类可以:

# 如上图所示处理索引和偏移量
# 将特征窗口拆分成一(features, labels)对
# 绘制结果窗口的内容
# 使用tf.data.Datasets从训练、评估和测试数据高效地生成这些窗口的批次


# 1, 索引和偏移量
# 从创建WindowGenerator类开始. 该__init__方法包括输入和标签索引的所有必要逻辑
# 它还将训练, 评估和测试数据帧作为输入. 这些将在tf.data.Dataset稍后转换为windows 的 s

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


# 下面是创建本节开头图表中显示的 2 个窗口的代码:
w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=['T (degC)'])
print(w1)

w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['T (degC)'])
print(w2)


# 2, 分裂
# 给定一个列表连续输入, 该split_window方法会将它们转换为输入窗口和标签窗口
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window

# 试试看:
# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')


# 通常, TensorFlow 中的数据被打包到数组中, 其中最外层的索引跨示例（ "批量" 维度）
# 中间索引是 "时间" 或 "空间"（宽度、高度）维度. 最里面的索引是特征

# 上面的代码采用了一批 3 个, 7 个时间步长的窗口, 每个时间步长有 19 个特征
# 它将它们分成一批 6 时间步长, 19 个特征输入和一个 1 时间步长 1 特征标签
# 标签只有一个特征, 因为WindowGenerator是用 初始化的label_columns=['T (degC)']. 本教程最初将构建预测单个输出标签的模型

# 3, 阴谋
# 这是一个绘图方法, 它允许对拆分窗口进行简单的可视化:
w2.example = example_inputs, example_labels

def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')

WindowGenerator.plot = plot

# 此图根据项目引用的时间对齐输入, 标签和（稍后的）预测:
w2.plot()
plt.show()

# 您可以绘制其他列, 但示例窗口w2配置只有T (degC)列的标签
w2.plot(plot_col='p (mbar)')
plt.show()


# 4, 创建stf.data.Dataset
# 最后, 此make_dataset方法将采用时间序列DataFrame并使用该函数将其转换tf.data.Dataset为(input_window, label_window)一对对preprocessing.timeseries_dataset_from_array
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)
    return ds

WindowGenerator.make_dataset = make_dataset


# 该WindowGenerator对象包含训练, 验证和测试数据。
# 添加属性以tf.data.Datasets使用上述make_dataset方法访问它们. 还添加一个标准示例批次, 以便于访问和绘图:
@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# 现在WindowGenerator对象允许您访问tf.data.Dataset对象, 因此您可以轻松地遍历数据
# 该Dataset.element_spec属性告诉您dtypes数据集元素的结构和形状

# Each element is an (inputs, label) pair
print(w2.train.element_spec)

# 迭代 aDataset产生具体批次
for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
    
    
''' 单步模型 '''
# 您可以基于此类数据构建的最简单模型是仅基于当前条件预测未来 1 个时间步 (1h) 的单个特征值的模型

# 因此, 首先要构建模型来预测T (degC)未来 1 小时的值
# 配置一个WindowGenerator对象以生成这些单步(input, label)对:
single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=['T (degC)'])
print(single_step_window)

# 该window对象tf.data.Datasets从训练 验证和测试集创建, 使您可以轻松地迭代批量数据
for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
    

### 基线 ###
# 在构建可训练模型之前, 最好将性能基线作为与后来更复杂模型进行比较的点
# 第一个任务是在给定所有特征的当前值的情况下预测未来 1 小时的温度. 当前值包括当前温度
# 所以从一个只返回当前温度作为预测的模型开始, 预测 "没有变化"
# 这是一个合理的基线, 因为温度变化缓慢. 当然, 如果您将来进一步进行预测, 则此基线的效果会较差
class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
# 实例化并评估此模型:
baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

# 这打印了一些性能指标, 但这些指标并不能让您了解模型的表现
# 该WindowGenerator有一个情节的方法, 但该地块不会只有一个样品是非常有趣的.
# 因此, 创建一个更宽的WindowGenerator窗口, 一次生成连续输入和标签的 24 小时窗口
# 在wide_window不改变模式的运作方式. 该模型仍然基于单个输入时间步长对未来 1 小时进行预测
# 在这里, time轴的作用类似于batch轴: 每个预测都是独立进行的, 时间步之间没有交互作用
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=['T (degC)'])
print(wide_window)

# 这个扩展的窗口可以直接传递给同一个baseline模型, 无需任何代码更改
# 这是可能的, 因为输入和标签具有相同的时间步数, 而基线只是将输入转发到输出:
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

# 绘制基线模型的预测, 您可以看到它只是标签, 向右移动了 1 小时
wide_window.plot(baseline)
plt.show()

# 在上面三个示例的图中, 单步模型运行了 24 小时. 这值得一些解释:
    # 蓝色的 "输入" 线显示了每个时间步的输入温度. 该模型接收所有特征, 该图仅显示温度
    # 绿色 "标签" 点显示目标预测值. 这些点显示在预测时间, 而不是输入时间. 这就是标签范围相对于输入移动 1 步的原因
    # 橙色的 "预测" 十字是模型对每个输出时间步长的预测. 如果模型预测完美, 预测将直接落在 "标签" 上
    
    
### 线性模型 ###
# 您可以应用于此任务的最简单的可训练模型是在输入和输出之间插入线性变换. 在这种情况下, 时间步的输出仅取决于该步:
# layers.Dense没有activation集合的A是线性模型
# 该层仅将数据的最后一个轴从 (batch, time, inputs)转换为(batch, time, units), 它独立应用于batch和time轴上的每个项目
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

# 本教程训练了很多模型, 因此将训练过程打包成一个函数:
MAX_EPOCHS = 20
def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
    return history

# 训练模型并评估其性能:
history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

# 与baseline模型一样, 线性模型可以在宽窗口的批次上调用. 使用这种方式, 模型对连续的时间步长进行一组独立的预测
# 该time轴的作用类似于另一个batch轴. 每个时间步的预测之间没有交互作用
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

# 这是它在 上的示例预测图wide_window, 请注意, 在许多情况下, 预测显然比仅返回输入温度要好, 但在某些情况下, 情况更糟:

# 线性模型的一个优点是它们相对容易解释. 您可以拉出图层的权重, 并查看分配给每个输入的权重:
plt.bar(x = range(len(train_df.columns)), height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()

# 有时模型甚至没有将最大的权重放在 input 上T (degC). 这是随机初始化的风险之一

### 稠密 ###
# 在应用实际操作多个时间步的模型之前, 有必要检查更深入,更强大的单输入步模型的性能
# 这是一个类似于linear模型的模型, 除了它Dense在输入和输出之间堆叠了几个层:
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)


### 多步密集 ###
# 单时间步模型没有其输入的当前值的上下文. 
# 它看不到输入特征是如何随时间变化的. 为了解决这个问题, 模型在进行预测时需要访问多个时间步长:

# 基线,线性和密集模型独立处理每个时间步. 在这里,模型将采用多个时间步长作为输入以产生单个输出
# 创建一个WindowGenerator将生成 3 小时输入和 1 小时标签的批次:
# 请注意，Window的shift参数相对于两个窗口的末尾
CONV_WIDTH = 3
conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1, label_columns=['T (degC)'])
print(conv_window)

conv_window.plot()
plt.title("Given 3h as input, predict 1h into the future.")
plt.show()

# 您可以dense通过添加 alayers.Flatten作为模型的第一层在多输入步骤窗口上训练模型:
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)

# IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

conv_window.plot(multi_step_dense)
plt.show()

# 这种方法的主要缺点是生成的模型只能在这种形状的输入窗口上执行
print('Input shape:', wide_window.example[0].shape)
try:
  print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')

# 下一节中的卷积模型解决了这个问题


### 卷积神经网络 ###
# 卷积层 ( layers.Conv1D) 也将多个时间步长作为每个预测的输入
# 下面是与相同的模型multi_step_dense 用卷积重写
# 注意变化:
    # 在layers.Flatten与第一layers.Dense被一个取代layers.Conv1D
    # 该layers.Reshape自卷积保持在其输出时间轴不再是必要的
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=(CONV_WIDTH,), activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

# 在示例批次上运行它以查看模型生成具有预期形状的输出:
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

# 对它进行训练和评估conv_window, 它应该提供与multi_step_dense模型相似的性能
history = compile_and_fit(conv_model, conv_window)

# IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

# 这conv_model与multi_step_dense模型之间的区别在于conv_model可以在任何长度的输入上运行. 卷积层应用于输入的滑动窗口:
# 如果在更宽的输入上运行它, 它会产生更宽的输出:
print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

# 请注意, 输出比输入短. 为了使训练或绘图工作，您需要标签和预测具有相同的长度
# 因此, 构建一个WindowGenerator使用一些额外输入时间步长生成宽窗口, 以便标签和预测长度匹配:
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['T (degC)'])
print(wide_conv_window)

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)


# 现在, 您可以在更宽的窗口上绘制模型的预测. 
# 注意第一次预测之前的 3 个输入时间步长. 这里的每个预测都基于前面的 3 个时间步:
wide_conv_window.plot(conv_model)
plt.show()


### 循环神经网络 ###
# 循环神经网络 (RNN) 是一种非常适合时间序列数据的神经网络. RNN 逐步处理时间序列, 从时间步到时间步保持内部状态

# 有关更多详细信息, 请阅读文本生成教程或RNN 指南
# 在本教程中, 您将使用称为长短期记忆 ( LSTM )的 RNN 层
# 所有 keras RNN 层的一个重要构造函数参数是return_sequences参数. 此设置可以通过以下两种方式之一配置图层
    # 1, 如果False, 默认情况下, 该层仅返回最终时间步长的输出, 让模型有时间在进行单个预测之前预热其内部状态:
    # 2, 如果True该层为每个输入返回一个输出. 这对于：
        # 堆叠 RNN 层
        # 同时在多个时间步上训练模型
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# 使用return_sequences=True该模型可以一次对 24 小时的数据进行训练
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)

# IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

wide_window.plot(lstm_model)
plt.show()

### 表现 ###
# 有了这个数据集, 通常每个模型都比之前的模型做得稍微好一些
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
_ = plt.legend()
plt.show()

for name, value in performance.items():
    print(f'{name:12s}: {value[1]:0.4f}')
    
    
### 多输出型号 ###
# 到目前为止, 所有模型都预测T (degC)了单个时间步长的单个输出特征
# 所有这些模型都可以转换为预测多个特征, 只需更改输出层中的单元数并调整训练窗口以包含labels
single_step_window = WindowGenerator(
    # `WindowGenerator` returns all features as labels if you 
    # don't set the `label_columns` argument.
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(input_width=24, label_width=24, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
    
# 请注意, features标签的轴现在与输入具有相同的深度, 而不是 1

# 基线
# 这里可以使用相同的基线模型, 但这次重复所有特征而不是选择特定的label_index
baseline = Baseline()
baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(wide_window.val)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)

# 稠密
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(dense, single_step_window)

# IPython.display.clear_output()
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

# 循环神经网络
## %%time
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(lstm_model, wide_window)

# IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate( wide_window.val)
performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0)

print()

# 高级: 剩余连接
# 之前的Baseline模型利用了这样一个事实, 即序列不会随时间步长发生剧烈变化
# 到目前为止, 本教程中训练的每个模型都是随机初始化的, 然后必须了解输出与前一时间步长的微小变化
# 虽然您可以通过仔细的初始化来解决这个问题, 但将其构建到模型结构中会更简单
# 在时间序列分析中, 构建模型而不是预测下一个值, 而是预测值在下一个时间步长中的变化是很常见的
# 类似地, 深度学习中的 "残差网络" 或 "ResNets" 指的是每层都添加到模型累积结果的架构
# 这就是你如何利用变化应该很小的知识

# 本质上, 这会初始化模型以匹配Baseline. 对于这项任务, 它可以帮助模型更快地收敛, 性能稍好一些
# 这种方法可以与本教程中讨论的任何模型结合使用
# 这里是应用于 LSTM 模型, 注意使用tf.initializers.zeros来确保初始预测的变化很小, 并且不要压倒残差连接
# 这里的梯度没有对称性破坏问题, 因为zeros它们仅用于最后一层

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta
    
# # %%time
residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small
        # So initialize the output layer with zeros
        kernel_initializer=tf.initializers.zeros())
]))

history = compile_and_fit(residual_lstm, wide_window)

# IPython.display.clear_output()
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
print()

# 表现
# 以下是这些多输出模型的整体性能
x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
plt.ylabel('MAE (average over all outputs)')
_ = plt.legend()
plt.show()

for name, value in performance.items():
    print(f'{name:15s}: {value[1]:0.4f}')
    
# 上述性能是所有模型输出的平均值


''' 多步模型 '''
# 前几节中的单输出和多输出模型都对未来 1 小时进行了单时间步预测
# 本节介绍如何扩展这些模型以进行多时间步长预测
# 在多步预测中, 模型需要学习预测一系列未来值. 因此, 与仅预测单个未来点的单步模型不同, 多步模型预测未来值的序列
# 对此有两种粗略的方法:
    # 一次预测整个时间序列的单次预测
    # 自回归预测，其中模型仅进行单步预测，其输出作为输入反馈
# 在本节中, 所有模型将预测所有输出时间步长的所有特征
# 对于多步模型, 训练数据再次由每小时样本组成. 然而，在这里, 模型将学习预测未来的 24 小时, 给定过去的 24 小时
# 这是一个Window从数据集中生成这些切片的对象:
OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
plt.show()
print(multi_window)

#### 基线 #### 
# 此任务的一个简单基线是为所需的输出时间步数重复最后一个输入时间步长:
class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(last_baseline)
plt.show()

# 由于此任务是在给定 24 小时的情况下预测 24 小时, 因此另一种简单的方法是重复前一天, 假设明天类似:
class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(repeat_baseline)
plt.show()

### 单发型号 ###
# 解决此问题的一种高级方法是使用 "单次" 模型, 该模型在单个步骤中进行整个序列预测
# 这可以作为layers.Dense带OUT_STEPS*features输出单元有效地实现. 该模型只需要将该输出重塑为所需的(OUTPUT_STEPS, features)

# 线性
# 基于最后一个输入时间步长的简单线性模型比任一基线都好, 但动力不足
# 该模型需要OUTPUT_STEPS从具有线性投影的单个输入时间步长预测时间步长. 它只能捕获行为的低维切片, 可能主要基于一天中的时间和一年中的时间
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

# IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)
plt.show()


# 稠密
# layers.Dense在输入和输出之间添加 a 可以为线性模型提供更多功能, 但仍然仅基于单个输入时间步长
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

# IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model)
plt.show()


# CNN
# 卷积模型基于固定宽度的历史进行预测, 这可能会导致比密集模型更好的性能, 因为它可以看到事物随时间的变化:
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

# IPython.display.clear_output()
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)
plt.show()


# 循环神经网络
# 如果循环模型与模型所做的预测相关, 那么循环模型可以学习使用输入的悠久历史
# 在这里, 模型将累积 24 小时的内部状态, 然后对接下来的 24 小时进行单个预测
# 在这种单发格式中, LSTM 只需要在最后一个时间步产生一个输出,因此设置return_sequences=False
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

# IPython.display.clear_output()
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)
plt.show()


### 进阶：自回归模型 ###
# 上述模型都在一个步骤中预测了整个输出序列
# 在某些情况下, 模型将此预测分解为单独的时间步长可能会有所帮助
# 然后每个模型的输出可以在每一步反馈给自身, 并且可以根据前一个模型进行预测, 就像经典的用循环神经网络生成序列一样
# 这种类型的模型的一个明显优势是它可以设置为产生不同长度的输出
# 您可以采用本教程前半部分训练的任何单步多输出模型并在自回归反馈循环中运行, 但在这里您将专注于构建一个经过明确训练的模型

# 循环神经网络
# 本教程仅构建自回归 RNN 模型, 但此模式可应用于任何旨在输出单个时间步长的模型
# 该模型将具有与单步LSTM模型相同的基本形式: AnLSTM后跟layers.Dense将LSTM输出转换为模型预测的 a
# Alayers.LSTM是layers.LSTMCell包含在更高级别中的layers.RNN, 它为您管理状态和序列结果（有关详细信息, 请参阅Keras RNN）
# 在这种情况下, 模型必须手动管理每个步骤的输入, 因此它layers.LSTMCell直接用于较低级别的单时间步界面
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
        
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

# 该模型需要的第一个方法是warmup根据输入初始化其内部状态的方法. 
# 一旦经过训练, 此状态将捕获输入历史的相关部分. 这相当于之前的单步LSTM模型:
def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

FeedBack.warmup = warmup

# 此方法返回单个时间步长预测, 以及 LSTM 的内部状态:
prediction, state = feedback_model.warmup(multi_window.example[0])
print(prediction.shape)


# 使用RNN的状态和初始预测, 您现在可以继续迭代模型, 将每一步的预测作为输入反馈
# 收集输出预测的最简单方法是使用 python 列表, 并tf.stack在循环之后
def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the lstm state
    prediction, state = self.warmup(inputs)

    # Insert the first prediction
    predictions.append(prediction)

    # Run the rest of the prediction steps
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                                training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions

FeedBack.call = call

# 在示例输入上测试运行此模型:
print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

# 现在训练模型:
history = compile_and_fit(feedback_model, multi_window)

# IPython.display.clear_output()
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model)


### 表现 ###
# 在这个问题上, 作为模型复杂性的函数, 收益明显递减
x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show()

# 本教程前半部分的多输出模型指标显示了所有输出特征的平均性能. 这些性能相似, 但也是跨输出时间步长的平均值
for name, value in multi_performance.items():
    print(f'{name:8s}: {value[1]:0.4f}')
    
# 从密集模型到卷积和循环模型的收益只有几个百分点（如果有的话）, 而自回归模型的表现明显更差
# 所以这些更复杂的方法在这个问题上可能不值得, 但是没有尝试就没有办法知道, 这些模型可能对你的问题有帮助