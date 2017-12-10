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


def res_block(input, conv_filters: list, kernel_size=3, active='relu'):
    assert len(conv_filters) > 0
    output = input
    for filters in conv_filters:
        output = Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer=he_normal())(output)
        output = BatchNormalization()(output)
    output = Activation(active)(output)
    output2 = Conv2D(conv_filters[0], kernel_size=kernel_size, padding='same', kernel_initializer=he_normal())(input)
    output2 = BatchNormalization()(output2)
    output2 = Activation(active)(output2)
    output = keras.layers.concatenate([output, output2], axis=3)
    if K.image_data_format() == 'channels_first':
        input_channels = int(input.shape[1])
    else:
        input_channels = int(input.shape[3])
    output = Conv2D(input_channels, kernel_size=1, kernel_initializer=he_normal())(
        output)
    output = add([input, output])
    output = BatchNormalization()(output)
    output = Activation(active)(output)
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

output = res_block(output, [64, 64])
output = res_block(output, [128, 64])
output = AveragePooling2D()(output)
output = res_block(output, [256, 64])
output = AveragePooling2D()(output)
# output = res_block(output, [256, 256])
output = Flatten()(output)
output = Dense(512, activation='relu')(output)
output = Dropout(0.4)(output)
output = Dense(256, activation='relu')(output)
output = Dropout(0.4)(output)
output = Dense(class_num, activation='softmax')(output)

model = Model(inputs=input, outputs=output)
model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

checkpoint = ModelCheckpoint(filepath='cifa100res.h5',
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=0,
                               patience=3,
                               min_lr=0.00001)
tensorboard = TensorBoard(log_dir='/tmp/deeplogs/cifa100_resnet')
early_stop = EarlyStopping(patience=15)
callbacks = [checkpoint, lr_reducer, tensorboard, early_stop]
if os.path.exists('/tmp/deeplogs/cifa100_resnet'):
    rmtree('/tmp/deeplogs/cifa100_resnet')
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
                        workers=4,
                        epochs=100,
                        validation_data=(test_data, test_label),
                        callbacks=callbacks)

scores = model.evaluate(train_data, train_label, verbose=1)
print('Train loss:', scores[0])
print('Train accuracy:', scores[1])
scores = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
