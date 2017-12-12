import keras
from keras.callbacks import *
from keras import backend as K
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from shutil import rmtree

from cifa100 import cifar100 as utils
from keras.layers import *
from keras.models import *


def conv_2d(x, filters, strides=1, padding='valid', kernel_size=3):
    return Conv2D(filters, strides=strides, padding=padding, kernel_size=kernel_size, kernel_initializer=he_normal(),
                  kernel_regularizer=l2(0.0001))(x)


def inception_res_block(x, filters, padding='same', active='relu'):
    shrink = True
    if padding == 'same':
        shrink = False
    if shrink:
        x_shortcut = conv_2d(x, filters, strides=2)
    else:
        x_shortcut = conv_2d(x, filters, strides=1, padding='same')
    x_shortcut = BatchNormalization()(x_shortcut)
    x_shortcut = Activation(active)(x_shortcut)

    pool = conv_2d(x_shortcut, 64, kernel_size=1)
    pool = MaxPool2D(strides=1, padding='same')(pool)

    single_conv = conv_2d(x_shortcut, 64, padding='same')
    single_conv = BatchNormalization()(single_conv)
    single_conv = Activation(active)(single_conv)
    single_conv = conv_2d(single_conv, 64, kernel_size=1)
    single_conv = BatchNormalization()(single_conv)

    double_conv = conv_2d(x_shortcut, 64, padding='same')
    double_conv = BatchNormalization()(double_conv)
    double_conv = Activation(active)(double_conv)
    double_conv = conv_2d(double_conv, 128, padding='same')
    double_conv = BatchNormalization()(double_conv)
    double_conv = Activation(active)(double_conv)
    double_conv = conv_2d(double_conv, 64, kernel_size=1)
    double_conv = BatchNormalization()(double_conv)

    linear_conv = conv_2d(x_shortcut, 64, kernel_size=1)
    linear_conv = BatchNormalization()(linear_conv)
    linear_conv = Activation(active)(linear_conv)

    concat_output = concatenate([pool, single_conv, double_conv, linear_conv], axis=3)
    output = conv_2d(concat_output, filters, kernel_size=1)
    output = add([x_shortcut, output])
    output = BatchNormalization()(output)
    return output


class_num = 100
filters = 64
data_augmentation = True

(train_data, train_label), (test_data, test_label) = utils.load_data()
train_label = keras.utils.to_categorical(train_label, class_num)
test_label = keras.utils.to_categorical(test_label, class_num)

input = Input(train_data.shape[1:])
output = Conv2D(64, 3, padding='same', kernel_initializer=he_normal(), kernel_regularizer=l2(0.0001))(input)
output = BatchNormalization()(output)
output = Activation('relu')(output)

output = inception_res_block(output, 64)
output = inception_res_block(output, 64)
output = AveragePooling2D()(output)
output = inception_res_block(output, 128)
output = inception_res_block(output, 128)
output = AveragePooling2D()(output)
# output = res_block(output, [256, 256])
output = Flatten()(output)
output = Dense(256, activation='relu')(output)
output = Dropout(0.4)(output)
output = Dense(class_num, activation='softmax')(output)

model = Model(inputs=input, outputs=output)
if os.path.exists("cifa100_inception_res.h5"):
    model.load_weights("cifa100_inception_res.h5")
model.compile(optimizer=keras.optimizers.SGD(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

checkpoint = ModelCheckpoint(filepath='cifa100_inception_res.h5',
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=0,
                               patience=3,
                               min_lr=0.00001)
tensorboard = TensorBoard(log_dir='/tmp/deeplogs/cifa100_inception_resnet')
early_stop = EarlyStopping(patience=15)
callbacks = [checkpoint, lr_reducer, tensorboard, early_stop]
if os.path.exists('/tmp/deeplogs/cifa100_inception_resnet'):
    rmtree('/tmp/deeplogs/cifa100_inception_resnet')
if not data_augmentation:
    model.fit(train_data, train_label, batch_size=16, epochs=500, validation_data=(test_data, test_label),
              callbacks=callbacks)
else:
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    datagen.fit(train_data)
    model.fit_generator(datagen.flow(train_data, train_label, batch_size=16),
                        steps_per_epoch=train_data.shape[0] // 16,
                        workers=12,
                        epochs=500,
                        validation_data=(test_data, test_label),
                        callbacks=callbacks)

scores = model.evaluate(train_data, train_label, verbose=1)
print('Train loss:', scores[0])
print('Train accuracy:', scores[1])
scores = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
