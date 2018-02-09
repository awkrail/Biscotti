import argparse
import os
import glob

import numpy as np
np.random.seed(2016)
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
# from keras.layers import merge
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import list_pictures, array_to_img

def create_fcn(input_size):
    # tensorflowに書き換え
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
    pool5 = MaxPooling2D(pool_size=(3, 3), data_format="channels_last")(conv5) # 元は(2, 2)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv6)

    # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
    up7 = concatenate([UpSampling2D(size=(3, 3), data_format="channels_last")(conv6), conv5], axis=3) # 元は(2, 2)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(up7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv7)

    # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv4], mode='concat', concat_axis=1)
    up8 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv8)

    # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv9)

    # up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
    up10 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv10)

    # up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
    up11 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv11)

    conv12 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv11)

    fcn = Model(input=inputs, output=conv12)

    return fcn

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


if __name__ == "__main__":
    target_size = (1200, 1200) # 変更
    dname_checkpoints = 'checkpoints'
    dname_outputs = 'outputs'
    fname_architecture = 'architecture.json'
    fname_weights = "model_weights_{epoch:02d}.h5"
    fname_stats = 'stats.npz'
    dim_ordering = 'channels_last'

    # 画像のサイズを固定してやってみる
    train_path = glob.glob("train/*.npy")
    length = len(train_path)
    X = np.zeros((length, 1200, 1200, 3)) # TODO: 画像のサイズは固定じゃないようにしたい
    y = np.zeros((length, 1200, 1200, 3))

    for i, t_path in enumerate(train_path):
        data = np.load(t_path)
        img, dqt = data[:, :, :3], data[:, :, 3:]
        X[i] = img
        y[i] = dqt
    print("==> loaded!")
    print("creating model ...")

    X_train, Y_train = X[:400, :, :, :], y[:400,:, :, :]
    X_valid, Y_valid = X[400:, :, :, :], y[400:, :, :, :]
    model = create_fcn(target_size)

    # 損失関数，最適化手法を定義
    adam = Adam(lr=1e-5)
    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

    json_string = model.to_json()
    with open("architecture/architecture.json", 'w') as f:
        f.write(json_string)

    checkpointer = ModelCheckpoint(filepath="checkpoints/" + fname_weights, save_best_only=False)

    # トレーニングを開始
    print('start training...')
    model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1,
                shuffle=True, validation_data=(X_valid, Y_valid),
                callbacks=[checkpointer])






