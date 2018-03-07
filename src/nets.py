from keras.models import Model
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate, concatenate, LeakyReLU, BatchNormalization, Activation

def create_fcn(input_size):
    # EncoderSide
    inputs = Input((input_size[0], input_size[1], 3))

    conv1 = Conv2D(32, (3, 3), padding='same', data_format="channels_last")(inputs)
    conv1 = LeakyReLU(0.2)(conv1)

    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    # pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    # pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv3)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    # pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv4)
    conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    # pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv6 = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv5)
    conv6 = Conv2D(1024, (3, 3), padding='same', data_format="channels_last")(conv6)
    conv6 = Activation('relu')(conv6)

    # Decoder Side
    up1 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv5], axis=3) # 元は(2, 2)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(up1)
    conv7 = BatchNormalization(axis=-1)(conv7)

    up2 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up2)
    conv8 = BatchNormalization(axis=-1)(conv8)

    up3 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up3)
    conv9 = BatchNormalization(axis=-1)(conv9)

    up4 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up4)
    conv10 = BatchNormalization(axis=-1)(conv10)

    up5 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up5)
    conv11 = BatchNormalization(axis=-1)(conv11)

    conv12 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv11)
    fcn = Model(input=inputs, output=conv12)

    return fcn


def my_load_generator(img_shape):
    model = create_fcn(img_shape)
    model.summary()
    return model