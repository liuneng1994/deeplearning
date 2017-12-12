import tensorflow as tf

from cifar100_tensorflow.dataset_utils import Cifar100


def conv_2d(x, filters, strides=1, padding='valid', kernel_size=3, l2=0.0001):
    return tf.layers.Conv2D(filters, strides=strides, padding=padding, kernel_size=kernel_size,
                            kernel_initializer=tf.keras.initializers.he_normal(),
                            kernel_regularizer=tf.keras.regularizers.l2(l2))(x)


def inception_res_block(x, filters, blocks, padding='same', active=tf.nn.relu, training=True):
    with tf.name_scope("inception_res_block_%d" % blocks):
        shrink = True
        if padding == 'same':
            shrink = False
        with tf.name_scope("shortcut"):
            if shrink:
                x_shortcut = conv_2d(x, filters, strides=2)
            else:
                x_shortcut = conv_2d(x, filters, strides=1, padding='same')
            x_shortcut = tf.layers.BatchNormalization()(x_shortcut, training=training)
            x_shortcut = active(x_shortcut)
        with tf.name_scope("pooling"):
            pool = conv_2d(x_shortcut, 64, kernel_size=1)
            pool = tf.layers.MaxPooling2D((2, 2), strides=1, padding='same')(pool)

        with tf.name_scope("single_conv"):
            single_conv = conv_2d(x_shortcut, 64, padding='same')
            single_conv = tf.layers.BatchNormalization()(single_conv, training=training)
            single_conv = active(single_conv)
            single_conv = conv_2d(single_conv, 64, kernel_size=1)
            single_conv = tf.layers.BatchNormalization()(single_conv, training=training)

        with tf.name_scope("double_conv"):
            double_conv = conv_2d(x_shortcut, 64, padding='same')
            double_conv = tf.layers.BatchNormalization()(double_conv, training=training)
            double_conv = active(double_conv)
            double_conv = conv_2d(double_conv, 128, padding='same')
            double_conv = tf.layers.BatchNormalization()(double_conv, training=training)
            double_conv = active(double_conv)
            double_conv = conv_2d(double_conv, 64, kernel_size=1)
            double_conv = tf.layers.BatchNormalization()(double_conv, training=training)

        with tf.name_scope("linear_conv"):
            linear_conv = conv_2d(x_shortcut, 64, kernel_size=1)
            linear_conv = tf.layers.BatchNormalization()(linear_conv, training=training)
            linear_conv = active(linear_conv)
        with tf.name_scope("concat_filter"):
            concat_output = tf.concat([pool, single_conv, double_conv, linear_conv], axis=3)
            output = conv_2d(concat_output, filters, kernel_size=1)

        with tf.name_scope("residual_add"):
            output = tf.add(x_shortcut, output)
        return output


class HParam:
    def __init__(self, classes, keep_prop=0.4, lr=0.01, batch_size=16, epochs=100, data_augmentation=True):
        self.learning_rate = lr
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.classes = classes
        self.keep_prop = keep_prop
        self.epochs = epochs


class InceptionResnet:
    def __init__(self, hparam: HParam, training):
        assert isinstance(hparam, HParam)
        assert isinstance(training, bool)
        self.hparam = hparam
        self.training = training
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
        y = tf.placeholder(tf.int32, shape=[None, 1], name="y")
        self.x = x
        self.y = y
        with tf.name_scope("first_conv"):
            output = conv_2d(x, filters=64, kernel_size=3, padding='same')
            output = tf.layers.BatchNormalization()(output, training=self.training)
            output = tf.nn.relu(output)
        output = inception_res_block(output, 64, 1, training=self.training)
        output = inception_res_block(output, 64, 2, padding='valid', training=self.training)
        output = inception_res_block(output, 128, 3, training=self.training)
        output = inception_res_block(output, 128, 3, training=self.training)
        output = tf.layers.AveragePooling2D(pool_size=2, strides=2)(output)
        with tf.name_scope("fully_connect"):
            logits = tf.layers.Flatten()(output)
            logits = tf.layers.Dense(512, activation=tf.nn.relu)(logits)
            logits = tf.layers.Dropout(self.hparam.keep_prop)(logits, training=self.training)
            logits = tf.layers.Dense(256, activation=tf.nn.relu)(logits)
            logits = tf.layers.Dropout(self.hparam.keep_prop)(logits, training=self.training)
            logits = tf.layers.Dense(self.hparam.classes)(logits)
        self.encode = output
        self.logits = logits
        self.losses = tf.losses.sparse_softmax_cross_entropy(y, logits)
        self.predict = tf.argmax(tf.nn.softmax(self.logits), axis=1, output_type=tf.int32)
        self.accuracy = tf.metrics.accuracy(y, self.predict)
        if self.training:
            self.lr = tf.Variable(self.hparam.learning_rate, name="learning_rate", trainable=False)
            self.global_steps = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            self.step = self.optimizer.minimize(loss=self.losses, global_step=self.global_steps)
            with tf.name_scope("train"):
                train_acc = tf.summary.scalar("acc", self.accuracy)
                train_loss = tf.summary.scalar("loss", self.losses)
            self.summary_op = tf.summary.merge([train_acc, train_loss])
