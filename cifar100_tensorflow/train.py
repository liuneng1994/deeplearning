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


def update_lr(epoch, sess):
    if epoch > 0:
        sess.run(tf.assign(model.lr, 0.01))
    elif epoch > 20:
        sess.run(tf.assign(model.lr, 0.001))
    elif epoch > 40:
        sess.run(tf.assign(model.lr, 0.0001))


with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/deeplearning/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(model.hparam.epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                x, y = sess.run(iterator.get_next())
                _, summary, loss, acc, steps = sess.run(
                    [model.step, model.summary_op, model.losses, model.accuracy, model.global_steps],
                    feed_dict={model.x: x, model.y: y})
                tf.logging.log_every_n('debug', "step %5d: losses %.5f ,acc %.5f" % (loss, acc, steps), 100)
                writer.add_summary(summary, steps)
            except tf.errors.OutOfRangeError:
                break
        loss, acc = sess.run([model.losses, model.accuracy],
                             feed_dict={model.x: data.test_data, model.y: data.test_fine_label})
        tf.logging.info("epoch %5d: losses %.5f ,acc %.5f" % (loss, acc, epoch))
        update_lr(epoch, sess)

        for name, value in [('val_loss', loss), ('val_acc', acc)]:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            writer.add_summary(summary)
        writer.flush()

if __name__ == '__main__':
    tf.app.run()
