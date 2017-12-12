import numpy as np
import os
from keras.utils import data_utils
import pickle


def unpickle(file: str):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def _load_data(basedir='.'):
    meta, train, test = (unpickle(os.path.join(basedir, "meta")), unpickle(os.path.join(basedir, "train")),
                         unpickle(os.path.join(basedir, "test")))
    train_data = train[b'data'].reshape((-1, 3, 32, 32))
    test_data = test[b'data'].reshape((-1, 3, 32, 32))
    train_fine_label = np.asarray(train[b'fine_labels']).reshape((-1, 1))
    train_coarse_label = np.asarray(train[b'coarse_labels']).reshape((-1, 1))
    test_fine_label = np.asarray(test[b'fine_labels']).reshape((-1, 1))
    test_coarse_label = np.asarray(test[b'coarse_labels']).reshape((-1, 1))
    fine_label_names = meta[b'fine_label_names']
    coarse_label_names = meta[b'coarse_label_names']
    return (
        train_data, train_fine_label, train_coarse_label, test_data, test_fine_label, test_coarse_label,
        fine_label_names,
        coarse_label_names)


class Cifar100:
    def __init__(self):
        dir_name = 'cifar-100-python'
        origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        path = data_utils.get_file(dir_name, origin=origin, untar=True, cache_subdir='deep')
        train_data, train_fine_label, train_coarse_label, test_data, test_fine_label, test_coarse_label, \
        fine_label_names, coarse_label_names = _load_data(path)
        train_fine_label = np.reshape(train_fine_label, (len(train_fine_label), 1))
        train_coarse_label = np.reshape(train_coarse_label, (len(train_coarse_label), 1))
        test_fine_label = np.reshape(test_fine_label, (len(test_fine_label), 1))
        test_coarse_label = np.reshape(test_coarse_label, (len(test_coarse_label), 1))
        self.train_data = train_data
        self.train_fine_label = train_fine_label
        self.train_coarse_label = train_coarse_label
        self.test_data = test_data
        self.test_fine_data = test_fine_label
        self.test_coarse_label = test_coarse_label
        self.fine_label_names = fine_label_names
        self.coarse_label_names = coarse_label_names

    def decode_label(self, label_indexes, label_type="fine"):
        """
        decode label indexes to label names
        :param label_indexes: label indexes, type int or list
        :param label_type: label type, 'fine' or 'coarse'
        :return: label names, type is same to a label_indexes
        """
        names = None
        if label_type == 'fine':
            assert self.fine_label_names is not None
            names = self.fine_label_names
        elif label_type == 'coarse':
            assert self.coarse_label_names is not None
            names = self.coarse_label_names
        else:
            raise ValueError("label_type %s is illegal, should be fine or coarse" % label_type)
        if isinstance(label_indexes,list):
            results = []
            for index in label_indexes:
                results.append(names[index])
            return results
        elif isinstance(label_indexes,int):
            return names[label_indexes]
        else:
            raise TypeError("label_indexes type should be list or int, but is %s", str(type(label_indexes)))
