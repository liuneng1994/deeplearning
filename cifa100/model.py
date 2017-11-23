import keras
from cifa100 import cifar100 as utils
from keras.layers import *
from keras.models import *

class_num = 100
(train_data, train_label), (test_data, test_label) = utils.load_data()
train_label = keras.utils.to_categorical(train_label,class_num)
test_label = keras.utils.to_categorical(test_label,class_num)

model = Sequential()
model.add(
    Conv2D(32, (3, 3), padding='same', kernel_initializer=keras.initializers.he_normal(),
           input_shape=train_data.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), kernel_initializer=keras.initializers.he_normal()))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(64, (3, 3), padding='same', kernel_initializer=keras.initializers.he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer=keras.initializers.he_normal()))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer=keras.initializers.he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),kernel_initializer=keras.initializers.he_normal()))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_initializer=keras.initializers.he_normal()))
model.add(Dropout(0.5))
model.add(Dense(class_num, activation='softmax',kernel_initializer=keras.initializers.he_normal()))

model.compile(keras.optimizers.SGD(lr=0.1,decay=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

model.fit(train_data, train_label,epochs=200, validation_data=(test_data, test_label))

model.save("cifa100.h5")

scores = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
