import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)


''' 基本力学 '''
# Dataset 对象是一个 Python 可迭代对象。这使得使用 for 循环使用其元素成为可能
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

print(dataset)
for elem in dataset:
    print(elem.numpy())
    
# 或者通过使用 iter 显式创建 Python 迭代器并使用 next 使用其元素：
it = iter(dataset)

print(next(it).numpy())

# 或者，可以使用 reduce 转换来使用数据集元素，这会减少所有元素以产生单个结果。以下示例说明了如何使用 reduce 转换来计算整数数据集的总和。
print(dataset.reduce(0, lambda state, value: state + value).numpy())

# Dataset.element_spec 属性允许您检查每个元素组件的类型。
# 该属性返回 tf.TypeSpec 对象的嵌套结构，匹配元素的结构，可以是单个组件、组件元组或组件的嵌套元组。例如：
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]), tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.element_spec)

# 包含稀疏张量的数据集。
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
print(dataset4.element_spec)

# 使用 value_type 查看元素规范所代表的值的类型
print(dataset4.element_spec.value_type)


# 数据集转换支持任何结构的数据集。当使用 Dataset.map() 和 Dataset.filter() 转换时，将函数应用于每个元素，元素结构确定函数的参数：
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
print(dataset1)
for z in dataset1: print(z.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]), tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3)

for a, (b,c) in dataset3:
    print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))


''' 读取输入数据 '''

### 使用 NumPy 数组 ###
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset)

### 使用 Python 生成器 ###

# 另一个可以作为 tf.data.Dataset 轻松摄取的常见数据源是 python 生成器
def count(stop):
    i = 0
    while i<stop:
        yield i
        i += 1

for n in count(5):
    print(n)
    
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (),)

for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())
    
# output_shapes 参数不是必需的，但强烈建议使用，因为许多 TensorFlow 操作不支持具有未知等级的张量。
# 如果特定轴的长度未知或可变，请在 output_shapes 中将其设置为 None。
# 同样重要的是要注意 output_shapes 和 output_types 遵循与其他数据集方法相同的嵌套规则。
# 这是一个演示这两个方面的示例生成器，它返回数组的元组，其中第二个数组是一个长度未知的向量。
def gen_series():
    i = 0
    while True:
        size = np.random.randint(0, 10)
        yield i, np.random.normal(size=(size,))
        i += 1
        
for i, series in gen_series():
    print(i, ":", str(series))
    if i > 5:
        break
    
# 第一个输出是 int32，第二个是 float32。
# 第一项是标量，shape()，第二项是未知长度的向量，shape(None,)
ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes=((), (None,)))

print(ds_series)

# 现在它可以像普通的 tf.data.Dataset 一样使用。请注意，在对具有可变形状的数据集进行批处理时，您需要使用 Dataset.padded_batch
ds_series_batch = ds_series.shuffle(20).padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())


### 使用 TFRecord 数据 ###

# tf.data API 支持多种文件格式，因此您可以处理无法放入内存的大型数据集。
# 例如，TFRecord 文件格式是一种简单的面向记录的二进制格式，许多 TensorFlow 应用程序都使用它来训练数据。 
# tf.data.TFRecordDataset 类使您能够将一个或多个 TFRecord 文件的内容作为输入管道的一部分进行流式传输。
# 下面是一个使用来自法国街道名称标志 (FSNS) 的测试文件的示例。
# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

# TFRecordDataset 初始化程序的文件名参数可以是字符串、字符串列表或字符串的 tf.Tensor。
# 因此，如果您有两组文件用于训练和验证目的，您可以创建一个生成数据集的工厂方法，将文件名作为输入参数：
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
print(dataset)

# 许多 TensorFlow 项目在其 TFRecord 文件中使用序列化的 tf.train.Example 记录。这些需要在被检查之前被解码：
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())
print(parsed.features.feature['image/text'])


### 使用文本数据 ###

# 许多数据集作为一个或多个文本文件分发。 
# tf.data.TextLineDataset 提供了一种从一个或多个文本文件中提取行的简单方法。
# 给定一个或多个文件名，TextLineDataset 将为这些文件的每一行生成一个字符串值元素。

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)

# 这是第一个文件的前几行：
for line in dataset.take(5):
    print(line.numpy())

# 要在文件之间交替行，请使用 Dataset.interleave。
# 这使得将文件混在一起更容易。以下是每个翻译的第一行、第二行和第三行：

files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
    if i % 3 == 0:
        print()
    print(line.numpy())

# 默认情况下，TextLineDataset 生成每个文件的每一行，这可能是不可取的，例如，如果文件以标题行开头或包含注释。
# 可以使用 Dataset.skip() 或 Dataset.filter() 转换删除这些行。在这里，您跳过第一行，然后过滤以仅查找幸存者。
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)
for line in titanic_lines.take(10):
    print(line.numpy())

def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")

survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
    print(line.numpy())


### 使用 CSV 数据 ###

# 有关更多示例，请参阅加载 CSV 文件和加载 Pandas DataFrame。
# CSV 文件格式是一种以纯文本形式存储表格数据的流行格式。
# 例如：

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
df = pd.read_csv(titanic_file)
print(df.head())

# 如果您的数据适合内存，则相同的 Dataset.from_tensor_slices 方法适用于字典，从而可以轻松导入此数据：
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

# 一种更具可扩展性的方法是根据需要从磁盘加载。
# tf.data 模块提供了从一个或多个符合 RFC 4180 的 CSV 文件中提取记录的方法。
# Experimental.make_csv_dataset 函数是用于读取 csv 文件集的高级接口。它支持列类型推断和许多其他功能，例如批处理和混洗，以简化使用。
titanic_batches = tf.data.experimental.make_csv_dataset(titanic_file, batch_size=4, label_name="survived")

for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    print("features:")
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

# 如果您只需要列的子集，则可以使用 select_columns 参数。
titanic_batches = tf.data.experimental.make_csv_dataset(titanic_file, batch_size=4, label_name="survived", select_columns=['class', 'fare', 'survived'])

for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

# 还有一个较低级别的 experimental.CsvDataset 类，它提供了更细粒度的控制。它不支持列类型推断。相反，您必须指定每列的类型。

titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string] 
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

for line in dataset.take(10):
    print([item.numpy() for item in line])


### 使用文件集 ###

# 有许多数据集作为一组文件分布，其中每个文件都是一个示例。
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)

# 根目录包含每个类的目录：
for item in flowers_root.glob("*"):
    print(item.name)

# 每个类目录中的文件都是示例：
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

for f in list_ds.take(5):
    print(f.numpy())

# 使用 tf.io.read_file 函数读取数据并从路径中提取标签，返回 (image, label) 对：

def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label

labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())
