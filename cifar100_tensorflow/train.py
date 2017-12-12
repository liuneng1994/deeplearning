import numpy as np
import tensorflow as tf
from cifar100_tensorflow.model import InceptionResnet, HParam
from cifar100_tensorflow.dataset_utils import Cifar100

tf.flags.DEFINE_float("learning_rate", 0.01, "training learning rate")
tf.flags.DEFINE_integer("batch_size", 16, "training batch size")
tf.flags.DEFINE_integer("epochs", 100, "training epochs")
tf.flags.DEFINE_boolean("data_augmentation", True, "user data augmentation")
tf.flags.DEFINE_integer("classes", 100, "amount of object classes")
tf.flags.DEFINE_float("keep_prop", 0.4, "dropout keep prop rate")

flags = tf.flags.FLAGS
hparam = HParam(flags.classes, keep_prop=flags.keep_prop, lr=flags.learning_rate, batch_size=flags.batch_size,
                epochs=flags.epochs, data_augmentation=flags.data_augmentation)
model = InceptionResnet(hparam, training=True)

data = Cifar100()
data_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
    {'image': data.train_data, 'label': data.train_fine_label})
data_set: tf.data.Dataset = data_set.map(lambda record: (record['image'], record['label']))
data_set: tf.data.Dataset = data_set.batch(model.hparam.batch_size)
iterator = data_set.make_initializable_iterator()

writer = tf.summary.FileWriter("/tmp/deeplearning/")

with tf.Session() as sess:
    for _ in range(model.hparam.epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                data, label = sess.run(iterator.get_next())
                sess.run([model.step, model.summary_op, model.losses, model.accuracy, model.global_steps],
                         feed_dict={model.x: data, model.y: label})
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    tf.app.run()
