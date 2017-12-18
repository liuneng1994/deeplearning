from cifa100.cifar100 import load_data
import numpy as np
import datetime

(train_data, train_label), (test_data, test_label) = load_data()
batches = np.split(train_data, 32, 0)
start = datetime.datetime.now()
for batch in batches:
    batch.shape
end = datetime.datetime.now()
print("cost %d ms" % (end - start).microseconds)
