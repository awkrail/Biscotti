import argparse
import os
import numpy as np
import pandas as pd
import subprocess

import keras.backend as K
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.callbacks import ModelCheckpoint

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


def generator_loss(y_true, y_pred):
  """
  独自の誤差関数の定義
  1. binary_crossentropy
  2. butteraugli score
  """
  # 1. binary_crossentropy
  cross_entropy_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
  # 2. butteraugli
  # DCT係数 => src/predictのようにguetzliに当てはまるようにcsvで保存 => guetzli_setter => butteraugliを計算する, は可能
  return cross_entropy_loss

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
    # generator_model = nets.get_generator(target_size)
    generator_model = nets.generator_butteraugli(target_size)
    # generator_model.compile(loss='binary_crossentropy', optimizer=opt_unet, metrics=['accuracy'])
    generator_model.compile(loss=generator_loss, optimizer=opt_unet)

    # checkpoint
    checkpointer = ModelCheckpoint(filepath=output + "/model_weights_{epoch:02d}.h5", save_best_only=False)

    # generator_model's first weights
    model_path = output + "train_tmp/models/model_weights_initial.h5"
    generator_model.save(model_path)

    # start training...
    for epoch in range(args.epoch):
        perms = np.random.permutation(len(train_files))
        perm_batch = [perms[i:i+batch_size] for i in range(0, len(train_files), batch_size)]
        progbar = generic_utils.Progbar(threshold)
        for i, pb in enumerate(perm_batch):
            X_train, y_train = load_train_data_on_batch(args.datasetpath, pb, train_files, batch_size)
            # TODO : add loss +butteraugli
            """
            1. X_train => 保存, train_tmp/raw_images以下に保存
            2. model_pathを受け取ってそれらをbiscotti
            3. butteraugliを train_tmp/raw_images/とtrain_tmp/predict_images/で比較する
            """
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