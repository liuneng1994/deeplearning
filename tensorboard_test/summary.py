"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.truncated_normal([784, 10]),name="w")
    b = tf.Variable(tf.zeros([10]),name="b")
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    optimizer = tf.train.GradientDescentOptimizer(0.1, )
    graph = tf.get_default_graph()
    gradients = optimizer.compute_gradients(loss=cross_entropy,var_list=(W,b))
    # 梯度可视化，可以看出梯度是否收敛，是否出现问题
    with tf.name_scope("train"):
        for gradient, tensor in gradients:
            tf.summary.histogram("%s_grad"%tensor.name, gradient,collections=["train_summary"])
    train_step = optimizer.apply_gradients(gradients,global_step=tf.train.get_global_step())

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tensor board
    writer = tf.summary.FileWriter('/tmp/mnist',graph=graph)

    # 根据collection来分别存储日志信息，从而实现train，test分开收集
    with tf.name_scope("train"):
        train_acc = tf.summary.scalar("acc", accuracy,collections=["train_summary"])
        train_loss = tf.summary.scalar("loss", cross_entropy,collections=["train_summary"])

    with tf.name_scope("test"):
        test_acc = tf.summary.scalar("acc", accuracy,collections=["test_summary"])
        test_loss = tf.summary.scalar("loss", cross_entropy,collections=["test_summary"])

    merge_op = tf.summary.merge_all("train_summary")
    test_merge_op = tf.summary.merge_all("test_summary")



    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for epoch in range(100):
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_summary, _ = sess.run([merge_op, train_step], feed_dict={x: batch_xs, y_: batch_ys})

        # 收集train信息
        train_summary, train_acc = sess.run([merge_op, accuracy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        writer.add_summary(train_summary, epoch)
        # Test trained model
        # 收集test信息
        test_summary, acc = sess.run([test_merge_op, accuracy], feed_dict={x: mnist.test.images,
                                                                      y_: mnist.test.labels})
        writer.add_summary(test_summary, epoch)
        print("train_acc: %f, test_acc:%f"%(train_acc,acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
