import keras
from keras.callbacks import *
from keras import backend as K
from cifa100 import cifar100 as utils
from keras.layers import *
from keras.models import *


def res_block(input,conv_filters:list,kernel_size=3,active='relu'):
    assert len(conv_filters) > 0
    output = None
    for filters in conv_filters:
        output = Conv2D(filters,kernel_size=kernel_size,padding='same')(input)
        output = BatchNormalization()(output)
        output = Activation(active)(output)
    if K.image_data_format() == 'channels_first':
        input_channels = input.shape[1]
    else:
        input_channels = input.shape[3]
    output = Conv2D(input_channels,kernel_size=input_channels)(output)
    add([input,output])
    output = BatchNormalization()(output)
    output = Activation(active)(output)
    return output

class_num = 100
filters = 64

(train_data, train_label), (test_data, test_label) = utils.load_data()
train_label = keras.utils.to_categorical(train_label,class_num)
test_label = keras.utils.to_categorical(test_label,class_num)

input = Input(train_data[1:])
output = Conv2D(64,3,padding='same')(input)
output = BatchNormalization()(output)
output = Activation('relu')(output)

output = res_block(output,[64,64])
output = MaxPool2D()(output)
output = res_block(output,[128,128])
output = MaxPool2D()(output)
output = res_block(output,[256,256])
output = Flatten()(output)
output = Dense(class_num,activation='softmax')(output)

model = Model(inputs=input,outputs=output)
model.compile(optimizer=keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255


checkpoint = ModelCheckpoint(filepath='cifa100cnn.h5',
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=0.01,
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