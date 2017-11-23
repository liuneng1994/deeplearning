import numpy as np
import os
from keras.utils import data_utils
from cifa100 import utils
from keras import backend as K


def load_data():
    dirname = 'cifar-100-bastches'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = data_utils.get_file(dirname,origin=origin, untar=True, cache_dir=os.path.join('~''.deeplearning'))
    x_train, y_train, _, x_test, y_test, _ = utils.load_data(path)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)