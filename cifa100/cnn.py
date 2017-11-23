import keras
from keras.callbacks import *

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
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(64, (3, 3), padding='same', kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(class_num, activation='softmax',kernel_initializer=keras.initializers.he_normal()))

model.compile(keras.optimizers.SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255


checkpoint = ModelCheckpoint(filepath='cifa100cnn.h5',
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=2,
                               min_lr=0.5e-6)
tensorboard = TensorBoard(log_dir='/tmp/deeplogs/cifa100')
early_stop = EarlyStopping(patience=5)
callbacks = [checkpoint, lr_reducer,tensorboard,early_stop]

model.fit(train_data, train_label,epochs=100, validation_data=(test_data, test_label), callbacks=callbacks)

scores = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
