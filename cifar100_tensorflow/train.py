import numpy as np
import tensorflow as tf
from cifar100_tensorflow.model import InceptionResnet, HParam
from cifar100_tensorflow.dataset_utils import Cifar100
import os
import datetime

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
data_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(( data.train_data, data.train_fine_label))
data_set: tf.data.Dataset = data_set.batch(model.hparam.batch_size)
iterator = data_set.make_initializable_iterator()
x,y = iterator.get_next()


def update_lr(epoch, sess):
    if epoch > 0:
        sess.run(tf.assign(model.lr, 0.01))
    elif epoch > 20:
        sess.run(tf.assign(model.lr, 0.001))
    elif epoch > 40:
        sess.run(tf.assign(model.lr, 0.0001))


writer = tf.summary.FileWriter("/tmp/deeplearning/inception_resnet", tf.get_default_graph())
os.makedirs("/tmp/deeplearning/inception_resnet",exist_ok=True)
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if os.path.exists("/tmp/deeplearning/inception_resnet/checkpoint.ckpt"):
        saver.restore(sess,"/tmp/deeplearning/inception_resnet/checkpoint.ckpt")
    for epoch in range(model.hparam.epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                input,output = sess.run([x,y])
                _, predict,loss, acc, steps = sess.run(
                    [model.step, model.predict, model.losses, model.accuracy, model.global_steps],
                    feed_dict={model.x: input, model.y: output})
                if steps % 100 == 0:
                    print("step %05d: losses is %.5f ,acc is %.5f" % (steps,loss, acc))
            except tf.errors.OutOfRangeError:
                break
        # saver.save(sess,"/tmp/deeplearning/inception_resnet/checkpoint.ckpt",epoch)
        summary, loss, acc = sess.run([model.summary_op,model.losses, model.accuracy],
                             feed_dict={model.x: data.train_data[0:1000,:,:,:], model.y: data.train_fine_label[0:1000]})
        print("train epoch %05d: losses %.5f ,acc %.5f" % (epoch,loss, acc))
        writer.add_summary(summary, epoch)
        summary, loss, acc = sess.run([model.test_summary_op,model.losses, model.accuracy],
                             feed_dict={model.x: data.test_data, model.y: data.test_fine_label})
        print("tes epoch %05d: losses %.5f ,acc %.5f" % (epoch,loss, acc))
        writer.add_summary(summary, epoch)
        update_lr(epoch, sess)

if __name__ == '__main__':
    tf.app.run()
