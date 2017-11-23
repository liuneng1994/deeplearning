import pickle
import numpy as np
import os


def unpickle(file: str):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_data(basedir='.'):
    meta, train, test = (unpickle(os.path.join(basedir,"meta")),unpickle(os.path.join(basedir,"train")),unpickle(os.path.join(basedir,"test")))
    train_data = train[b'data'].reshape((-1,3,32,32))
    test_data = test[b'data'].reshape((-1,3,32,32))
    train_fine_label = np.asarray(train[b'fine_labels']).reshape((-1,1))
    train_coarse_label = np.asarray(train[b'coarse_labels']).reshape((-1,1))
    test_fine_label = np.asarray(test[b'fine_labels']).reshape((-1,1))
    test_coarse_label = np.asarray(test[b'coarse_labels']).reshape((-1,1))
    return train_data,train_fine_label,train_coarse_label,test_data,test_fine_label,test_coarse_label


if __name__ == '__main__':
    print(unpickle("../dataset/meta"))
    print(unpickle("../dataset/train"))
    print(unpickle("../dataset/test"))
