import argparse
import os
import numpy as np
import pandas as pd
import subprocess

import keras.backend as K
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.callbacks import ModelCheckpoint

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

# ==== Training Loss Functions ====
def resize444to420(dct):
  row = dct.shape[0]
  col = dct.shape[1]

  f_row = int(row / 2)
  f_col = int(col / 2)
  foundation = np.zeros((f_row, f_col))
  for j in range(0, row, 2):
    for i in range(0, col, 2):
      canvas4 = np.zeros((2, 2))
      for k in range(2):
        for u in range(2):
          canvas4[k][u] = dct[j+k][i+u]
        
        if canvas4[0][0] == 1:
          foundation[j//2][i//2] = 1
        else:
          foundation[j//2][i//2] = 0
  return foundation

def ModifyCoeffsForGuetzliDataStruct(dct):
  pass

def threshold(coeff):
  if coeff >= 0.5:
    return 1
  else:
    return 0

def dump_csv(pred, batch_num):
  thresholder = np.vectorize(threshold)
  dct_binary = thresholder(pred)

  y = pred[0]
  cr = resize444to420(pred[1])
  cb = resize444to420(pred[2])

  # modify for guetzli
  y = ModifyCoeffsForGuetzliDataStruct(y)
  cr = ModifyCoeffsForGuetzliDataStruct(cr)
  cb = ModifyCoeffsForGuetzliDataStruct(cb)

  # coeffs DataFrame
  y_df = pd.DataFrame(y)
  cr_df = pd.DataFrame(cr)
  cb_df = pd.DataFrame(cb)

  # dump csv
  y_df.to_csv("train_tmp/train_dct_csv/y_" + str(batch_num) + ".csv", header=None, index=None)
  cr_df.to_csv("train_tmp/train_dct_csv/cr_" + str(batch_num) + ".csv", header=None, index=None)
  cb_df.to_csv("train_tmp/train_dct_csv/cb_" + str(batch_num) + ".csv", header=None, index=None)


def butteraugli_loss(batch_y_pred):
  batch_scores = []
  import ipdb; ipdb.set_trace()

  # y_predを利用して, src/predict.pyのようにDCTをCSVでダンプして作成する
  for i, y_pred in enumerate(batch_y_pred):
    dump_csv(y_pred, i)
  
  for i in range(batch_y_pred.shape[0]):
    try:
      # TODO : モデルのpathを選択できるようにC++側を変更
      guetzli_setter = ["train_bin/Release/guetzli_setter", 
                        "train_tmp/raw_images/" + str(i) + ".jpg", 
                        "train_tmp/predict_images/" + str(i) + ".jpg",
                        "train_tmp/train_dct_csv/y_" + str(i) + ".csv",
                        "train_tmp/train_dct_csv/cb_" + str(i) + ".csv",
                        "train_tmp/train_dct_csv/cr_" + str(i) + ".csv"]
      subprocess.check_call(guetzli_setter)
      butteraguli_command = ["train_bin/Release/butteraugli", guetzli_setter[0], guetzli_setter[1]]
      score = subprocess.check_output(butteraguli_command)
      score = float(score)
      batch_scores.append(score)
    except:
      pass
  butteraugli = sum(batch_scores) / len(batch_scores)
  return 0.0001 * butteraugli

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
  butteraugli = butteraugli_loss(y_pred)
  return cross_entropy_loss + butteraugli

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
            import ipdb; ipdb.set_trace()
            # 実装案として、ここでX_trainの画像を保存する => train_on_batchで保存したものと比較する
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