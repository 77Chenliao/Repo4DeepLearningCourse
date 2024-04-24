import numpy as np
import struct
from sklearn.preprocessing import OneHotEncoder


def read_labels(filename):
    with open(filename, 'rb') as file:
        magic, num = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
    return labels

def read_images(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
    return images

train_labels = read_labels('./data/train-labels-idx1-ubyte')
train_images = read_images('./data/train-images-idx3-ubyte')
test_labels = read_labels('./data/t10k-labels-idx1-ubyte')
test_images = read_images('./data/t10k-images-idx3-ubyte')

# 数据预处理
X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
X_test = test_images.reshape(test_images.shape[0], -1) / 255.0
y_train = train_labels.reshape(-1, 1)
y_test = test_labels.reshape(-1, 1)
y_train = OneHotEncoder().fit_transform(y_train).toarray()
y_test = OneHotEncoder().fit_transform(y_test).toarray()

# 保存数据
np.save('./data/X_train.npy', X_train)
np.save('./data/X_test.npy', X_test)
np.save('./data/y_train.npy', y_train)
np.save('./data/y_test.npy', y_test)


