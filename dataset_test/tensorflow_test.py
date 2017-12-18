from cifa100.cifar100 import load_data
import tensorflow as tf
import datetime

(train_data, train_label), (test_data, test_label) = load_data()
data_set: tf.data.Dataset = tf.data.Dataset.from_tensors(train_data)
data_set: tf.data.Dataset = data_set.batch(32)
iterator = data_set.make_initializable_iterator()
start = datetime.datetime.now()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            x= sess.run(iterator.get_next())
        except tf.errors.OutOfRangeError:
            break
end = datetime.datetime.now()
print("cost %d ms"%(end-start).microseconds)