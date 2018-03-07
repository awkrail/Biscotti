import argparse
import os
import numpy as np

import keras.backend as K
from keras.optimizers import Adam

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
    return X_train, X_valid, y_train, y_valid

def train(args):
    output = args.outputfile
    if not os.path.exists("./figure"):
        os.mkdir("./figure")


    # load data
    images, dcts, images_val, dcts_val = load_img_and_dct_data(args.datasetpath)
    print("train_image shape: ", images.shape)
    print("train_dct shape: ", dcts.shape)
    print("validation image shape: ", images_val.shape)
    print("validation dct shape: ", dcts_val.shape)

    img_shape = images.shape[-3:]
    patch_num = (img_shape[0] // args.patch_size) * (img_shape[1] // args.patch_size)
    disc_img_shape = (args.patch_size, args.patch_size, images.shape[-1])

    # set optimizer
    opt_dcgan = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = nets.my_load_generator(img_shape)
    generator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator, metrics=['accuracy'])

    # checkpoint
    checkpointer = ModelCheckpoint(filepath="checkpoints/model_weights_{epoch:02d}.h5", save_best_only=False)
    # start training...
    print('start training...')
    generator_model.fit(images, dcts, batch_size=10, epochs=20, verbose=1,
                shuffle=True, validation_data=(images_val, dcts_val),
                callbacks=[checkpointer])


def main():
    parser = argparse.ArgumentParser(description="Training pix2pix")
    parser.add_argument("--datasetpath", '-d', type=str, required=True)
    parser.add_argument("--outputfile", "-o", type=str, required=True)
    parser.add_argument("--patch_size", "-p", type=int, default=112)
    parser.add_argument("--batch_size", "-b", type=str, default=5)
    parser.add_argument("--epoch", type=int, default=400)
    args = parser.parse_args()
    K.set_image_data_format("channels_last")

    train(args)

if __name__ == "__main__":
    main()
