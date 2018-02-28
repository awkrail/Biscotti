import argparse
import os
import glob

import numpy as np
np.random.seed(2016)
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
# from keras.layers import merge
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import list_pictures, array_to_img
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout

def create_fcn(input_size):
    inputs = Input((input_size[0], input_size[1], 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv5)
    
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(up7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv9)
    
    up10 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv10)
    
    up11 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv11)
    
    conv12 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv11)

    fcn = Model(input=inputs, output=conv12)
    return fcn

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(224, 224, 3))) # フーム
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256)) # ここら辺もうちょっと増やしてもいいのでは.
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

if __name__ == "__main__":
    target_size = (224, 224) # 変更
    batch_size = 20
    # 画像のサイズを固定してやってみる
    train_path = glob.glob("train224/*.npy")
    length = len(train_path)
    X = np.zeros((length, 224, 224, 3)) # TODO: 画像のサイズは固定じゃないようにしたい
    y = np.zeros((length, 224, 224, 3))

    for i, t_path in enumerate(train_path):
        data = np.load(t_path)
        img, dqt = data[:, :, :3], data[:, :, 3:]
        X[i] = img
        y[i] = dqt
    print("==> loaded!")
    print("creating model ...")

    # 半分ずつにして, discriminator, generatorの学習に両方使う
    X_train, Y_train = X[:1100, :, :, :], y[:1100,:, :, :]
    X_valid, Y_valid = X[1100:, :, :, :], y[1100:, :, :, :]

    del X # for free
    del y

    # generator
    g = create_fcn(target_size)
    g_optim = Adam(lr=1e-5)

    # discriminator
    d = discriminator_model()
    d_optim = Adam(lr=1e-5)

    # dis-gen
    d_on_g = generator_containing_discriminator(g, d)
    g.compile(loss="binary_crossentropy", optimizer="SGD")
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    for epoch in range(20):
        for index in range(X_train.shape[0]//batch_size):
            image_batch = X_train[index*batch_size:(index+1)*batch_size]
            guetzli_batch = Y_train[index*batch_size:(index+1)*batch_size]
            generated_image = g.predict(image_batch, verbose=0)
            X = np.concatenate((guetzli_batch, generated_image))
            y = [1] * batch_size + [0] * batch_size

            d_loss = d.train_on_batch(X, y)
            d.trainable = False
            image_for_g_batch = X_valid[index*batch_size:(index+1)*batch_size]
            g_loss = d_on_g.train_on_batch(image_for_g_batch, [1] * batch_size) # ここどうしよう
            print("batch {} d_loss: {}".format(index, d_loss))
            d.trainable = True
            print("batch {} g_loss: {}".format(index, g_loss))

        g.save_weights('gan_model/generator{}.hd5'.format(epoch))
        d.save_weights('gan_model/discriminator{}.hd5'.format(epoch))