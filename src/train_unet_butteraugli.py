import argparse
import os
import numpy as np
import pandas as pd
import subprocess

import keras.backend as K
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.callbacks import ModelCheckpoint

# For new model considering butteraguli
from keras.layers.core import Flatten, Dense, Activation, Lambda
from keras.models import Model
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate, concatenate, LeakyReLU, BatchNormalization

import tensorflow as tf
import nets


def load_img_and_dct_data(dataset_path):
    files = os.listdir(dataset_path)
    X = np.zeros((len(files), 224, 224, 3))
    y = np.zeros((len(files), 224, 224, 3))

    for i, file in enumerate(files):
        data = np.load(dataset_path + "/" + file)
        img, dct = data[:, :, :3], data[:, :, 3:]
        X[i] = img
        y[i] = dct
    
    threshold = int(X.shape[0]*0.9)
    X_train, X_valid = X[:threshold], X[threshold:]
    y_train, y_valid = y[:threshold], y[threshold:]
    return X_train, X_train, y_valid, y_valid


def load_img_and_dct_data_on_batch(dataset_path, dataset_files):
    load_length = len(dataset_files)
    X = np.zeros((load_length, 224, 224, 3))
    y = np.zeros((load_length, 224, 224, 3))

    for i, file in enumerate(dataset_files):
        data = np.load(dataset_path + "/" + file)
        img, dct = data[:, :, :3], data[:, :, 3:]
        X[i] = img
        y[i] = dct
    
    return X, y


def load_train_data_on_batch(dataset_path, perm, train_files, batch_size):
    X = np.zeros((batch_size, 224, 224, 3))
    y = np.zeros((batch_size, 224, 224, 3))
    for i, p_num in enumerate(perm):
        data = np.load(dataset_path + "/" + train_files[p_num])
        img, dct = data[:, :, :3], data[:, :, 3:]
        X[i] = img
        y[i] = dct
    return X, y 


def load_validation_dataset(dataset_path, test_files):
    X = np.zeros((len(test_files), 224, 224, 3))
    y = np.zeros((len(test_files), 224, 224, 3))
    for i, test_file in enumerate(test_files):
        data = np.load(dataset_path + "/" + test_file)
        img, dct = data[:, :, :3], data[:, :, 3:]
        X[i] = img
        y[i] = dct
    return X, y


class ButteruagliModel(Model):
  def __init__(self, sub_inputs, sub_outputs, butteraugli):
    super().__init__(inputs=sub_inputs, outputs=sub_outputs)
    self.butteraugli = [butteraugli]
  
  @property
  def losses(self):
    losses = super(self.__class__, self).losses
    return losses + self.butteraugli


class GeneratorModel():
  model = None

  def __init__(self, input_shape):
    self.input_shape = input_shape
  
  def build(self):
    inputs = Input((self.input_shape[0], self.input_shape[1], 3))
    outputs = self.unet(inputs)
    return ButteruagliModel(sub_inputs=[inputs], sub_outputs=[outputs], butteraugli=-1)

  def unet(self, inputs):
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)

    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(conv1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)

    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(conv2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)

    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(conv3)
    conv4 = Conv2D(256, (3, 3), padding="same")(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder Side
    up1 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv4), conv3], axis=3)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = BatchNormalization(axis=-1)(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv5), conv2], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = BatchNormalization(axis=-1)(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv1], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = BatchNormalization(axis=-1)(conv7)

    conv8 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv7)
    # fcn = Model(input=inputs, output=conv8)
    return conv8

def get_butteraugli_loss(x_train, model_path):
  """
  1. X_train => 保存, train_tmp/raw_images以下に保存
  2. model_pathを受け取ってそれらをbiscotti
  3. butteraugliを train_tmp/raw_images/とtrain_tmp/predict_images/で比較する
  """
  pass


def train(args):
    # load data
    data_files = sorted(os.listdir(args.datasetpath))
    threshold = int(len(data_files)*0.9)
    train_files = data_files[:threshold]
    test_files = data_files[threshold:]
    X_valid, y_valid = load_validation_dataset(args.datasetpath, test_files)
    batch_size = args.batch_size
    output = args.outputfile
    
    # set optimizer
    opt_unet = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    target_size = (224, 224, 3)
    # generator_model = nets.generator_butteraugli(target_size)
    generator_model = GeneratorModel(target_size).build()
    # generator_model.compile(loss='binary_crossentropy', optimizer=opt_unet, metrics=['accuracy'])
    generator_model.compile(loss='binary_crossentropy', optimizer=opt_unet)
    import ipdb; ipdb.set_trace()
    # checkpoint
    checkpointer = ModelCheckpoint(filepath=output + "/model_weights_{epoch:02d}.h5", save_best_only=False)

    # generator_model's first weights
    model_path = "train_tmp/models/model_weights_initial.h5"
    generator_model.save(model_path)

    # start training...
    for epoch in range(args.epoch):
        perms = np.random.permutation(len(train_files))
        perm_batch = [perms[i:i+batch_size] for i in range(0, len(train_files), batch_size)]
        progbar = generic_utils.Progbar(threshold)
        for i, pb in enumerate(perm_batch):
            X_train, y_train = load_train_data_on_batch(args.datasetpath, pb, train_files, batch_size)
            # TODO : add loss +butteraugli
            generator_model.trainable = False
            butteraugli_loss = get_butteraugli_loss(X_train, model_path)
            generator_model.butteraugli = butteraugli_loss
            generator_model.trainable = True
            loss = generator_model.train_on_batch(X_train, y_train)
            model_path = np.array("train_tmp/models/model_weights_{}_epoch_{}.h5".format(epoch, i))
            generator_model.save(model_path)
            progbar.add(batch_size, values=[("loss", loss[0]), ("accuracy", loss[1])])

        score = generator_model.evaluate(X_valid, y_valid)
        print("epoch {} : loss: {} accuracy {}".format(epoch, score[0], score[1]))
        # TODO: to load keras.models.load_model, change save_weights into model.save 
        # generator_model.save_weights(output + "/model_weights_{}.h5".format(epoch))
        # TODO: 損失関数をbutteraugliに変更する
        generator_model.save(output + '/model_weights_{}.h5'.format(epoch))


def main():
    parser = argparse.ArgumentParser(description="Training Unet")
    parser.add_argument("--datasetpath", '-d', type=str, required=True)
    parser.add_argument("--outputfile", "-o", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=400)
    args = parser.parse_args()
    K.set_image_data_format("channels_last")

    train(args)

if __name__ == "__main__":
    main()