import argparse
import os
import numpy as np

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
    return X_train, X_valid, y_train, y_valid


def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X


def get_disc_batch(X_dct, X_input, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        X_disc = generator_model.predict(X_input)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
    else:
        X_disc = X_dct
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
    
    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc

def train(args):
    output = args.outputfile
    if not os.path.exists("./figure"):
        os.mkdir("./figure")

    # load data
    images, images_val, dcts,  dcts_val = load_img_and_dct_data(args.datasetpath)
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
    generator_model = nets.get_generator(img_shape)
    discriminator_model = nets.get_discriminator(img_shape, disc_img_shape, patch_num)
    generator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator, metrics=['accuracy'])
    discriminator_model.trainable = False

    dcgan_model = nets.get_GAN(generator_model, discriminator_model, img_shape, args.patch_size)

    loss = ['binary_crossentropy', 'binary_crossentropy']
    loss_weights = [1E1, 1]
    dcgan_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_discriminator)

    print("start training...")
    for epoch in range(args.epoch):
        perm = np.random.permutation(images.shape[0]) # 入力側
        X_dct = dcts[perm]
        X_input_images = images[perm]
        X_dctIter = [X_dct[i:i+args.batch_size] for i in range(0, images.shape[0], args.batch_size)]
        X_imageIter = [X_input_images[i:i+args.batch_size] for i in range(0, images.shape[0], args.batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_imageIter)*args.batch_size)
        for X_dct_batch, X_input_batch in zip(X_dctIter, X_imageIter):
            b_it += 1

            X_disc, y_disc = get_disc_batch(X_dct_batch, X_input_batch, generator_model, b_it, args.patch_size)
            raw_disc, _ = get_disc_batch(X_input_batch, X_input_batch, generator_model, 1, args.patch_size)
            disc_input = X_disc + raw_disc

            # update discriminator
            disc_loss = discriminator_model.train_on_batch(disc_input, y_disc)

            idx = np.random.choice(dcts.shape[0], args.batch_size)
            X_gen_target, X_gen = dcts[idx], images[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = dcgan_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            discriminator_model.trainable = True

            progbar.add(args.batch_size, values=[
                ("D logloss", disc_loss),
                ("G loss1", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

    """
    # checkpoint
    checkpointer = ModelCheckpoint(filepath=output + "/model_weights_{epoch:02d}.h5", save_best_only=False)
    # start training...
    print('start training...')
    generator_model.fit(images, dcts, batch_size=10, epochs=20, verbose=1,
                shuffle=True, validation_data=(images_val, dcts_val),
                callbacks=[checkpointer])
    """


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
