import numpy as np
import os
from keras.utils import data_utils
from keras import backend as K
import pickle

def unpickle(file: str):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def _load_data(basedir='.'):
    meta, train, test = (unpickle(os.path.join(basedir,"meta")),unpickle(os.path.join(basedir,"train")),unpickle(os.path.join(basedir,"test")))
    train_data = train[b'data'].reshape((-1,3,32,32))
    test_data = test[b'data'].reshape((-1,3,32,32))
    train_fine_label = np.asarray(train[b'fine_labels']).reshape((-1,1))
    train_coarse_label = np.asarray(train[b'coarse_labels']).reshape((-1,1))
    test_fine_label = np.asarray(test[b'fine_labels']).reshape((-1,1))
    test_coarse_label = np.asarray(test[b'coarse_labels']).reshape((-1,1))
    return train_data,train_fine_label,train_coarse_label,test_data,test_fine_label,test_coarse_label

def load_data():
    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = data_utils.get_file(dirname,origin=origin, untar=True, cache_subdir='deep')
    x_train, y_train, _, x_test, y_test, _ = _load_data(path)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)