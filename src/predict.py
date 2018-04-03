import argparse

import numpy as np
import pandas as pd
import cv2
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation
# from keras.layers import merge
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import list_pictures, array_to_img
import matplotlib.pyplot as plt


class DctCsvLoader(object):
    def __init__(self, guetzli_csv_path, target_size):
        self.csv_path = guetzli_csv_path
        self.dct = pd.read_csv(guetzli_csv_path, header=None).get_values()
        self.target_size = target_size
    
    def get_csv(self):
        label = self.dct_csv2numpy_probability()
        seq = int(label.shape[0] * (2/3))
        # TODO: I need to adjust these variable values for input image size
        width = self.target_size[1]
        height = self.target_size[0]

        coeff_y, coeff_cbcr = label[:seq], label[seq:]
        coeff_y = self.resize_coeff_to_img_matrix(coeff_y, width, height)
        coeff_cb = self.resize420to444(coeff_cbcr[:int(seq/4)], width, height)
        coeff_cr = self.resize420to444(coeff_cbcr[int(seq/4):], width, height)
        coeff3d = np.concatenate((coeff_y, coeff_cr, coeff_cb), axis=2)
        return coeff3d

    def dct_csv2numpy_probability(self):
        checker_0 = np.vectorize(self.check0)
        csv_numpy = pd.read_csv(self.csv_path, header=None).get_values()
        return checker_0(csv_numpy == 0)

    @staticmethod
    def resize_coeff_to_img_matrix(coeff, width, height):
        canvas = np.zeros((height, width))
        width_blocks = int(width / 8)
        height_blocks = int(height / 8)
        for block_y in range(height_blocks):
            for block_x in range(width_blocks):
                block_ix = width_blocks * block_y + block_x
                block = coeff[block_ix].reshape(8, 8)
                canvas[block_y*8:block_y*8+8, block_x*8:block_x*8+8] = block
        return canvas.reshape(height, width, 1)

    @staticmethod
    def check0(coeff):
        if not coeff:
            return 1
        else:
            return 0
    
    @staticmethod
    def resize420to444(coeff, width, height):
        canvas = np.zeros((height, width))
        width_blocks = int(width / 16)
        height_blocks = int(height / 16)
        for block_y in range(height_blocks):
            for block_x in range(width_blocks):
                canvas16 = np.zeros((16, 16)).astype(np.int32)
                block_ix = width_blocks * block_y + block_x
                block = coeff[block_ix].reshape(8, 8)
                for i in range(8):
                    for j in range(8):
                        dct22 = block[i][j] * np.ones((2, 2)).astype(np.int32)
                        canvas16[i*2:i*2+2, j*2:j*2+2] = dct22
                canvas[block_y*16:block_y*16+16, block_x*16:block_x*16+16] = canvas16
        return canvas.reshape(height, width, 1)


class Predictor():
    def __init__(self, target_size, image, threshold, model_path, result_png_path, csv_path, guetzli_csv_path):
        self.target_size = target_size
        self.image = image
        self.threshold = threshold
        self.result_png_path = result_png_path
        self.csv_path = csv_path
        # TODO: Change 3layer
        self.model = self.create_generator_3layer()
        self.model.load_weights(model_path)
        self.guetzli_dct = DctCsvLoader(guetzli_csv_path, target_size).get_csv()
        self.predict_dct = None
    
    def predict(self):
        row = self.target_size[0]
        col = self.target_size[1]
        dct_predict = self.model.predict(image.reshape(1, row, col, 3)).reshape(row, col, 3)
        self.predict_dct = dct_predict

    def eval(self):
        # Y accuracy
        thresholder = np.vectorize(self.change_threshold)
        dct_binary = thresholder(self.predict_dct, self.threshold)
        print("evaluate accuracy...")
        y_accuracy = np.sum(dct_binary[:, :, 0] == self.guetzli_dct[:, :, 0]) / (dct_binary.shape[0] * dct_binary.shape[1])
        print("Y: ", y_accuracy)
        # Cr accuracy
        cr_accuracy = np.sum(dct_binary[:, :, 1] == self.guetzli_dct[:, :, 1]) / (dct_binary.shape[0] * dct_binary.shape[1])
        print("Cr: ", cr_accuracy)
        # Cb accuracy
        cb_accuracy = np.sum(dct_binary[:, :, 2] == self.guetzli_dct[:, :, 2]) / (dct_binary.shape[0] * dct_binary.shape[1])
        print("Cb: ", cb_accuracy)

    def plot(self, plot_guetzli_dct=True, binary=False):
        paths = ["y.png", "cr.png", "cb.png"]
        if plot_guetzli_dct:
            for i in range(3):
                self.plot_heatmap(self.guetzli_dct[:, :, i], self.result_png_path + "guetzli_" + paths[i])
        if binary:
            thresholder = np.vectorize(self.change_threshold)
            dct_binary = thresholder(self.predict_dct, self.threshold)
            for i in range(3):
                self.plot_heatmap(dct_binary[:, :, i], self.result_png_path + "binary_" + paths[i])
        for i in range(3):
            self.plot_heatmap(self.predict_dct[:, :, i], self.result_png_path + paths[i])
    
    def dump_csv(self):
        thresholder = np.vectorize(self.change_threshold)
        dct_binary = thresholder(self.predict_dct, self.threshold)

        y420 = dct_binary[:, :, 0]
        cr420 = self.resize444to420(dct_binary[:, :, 1])
        cb420 = self.resize444to420(dct_binary[:, :, 2])

        # modify for guetzli
        y = self.ModifyCoeffsForGuetzliDataStruct(y420)
        cr = self.ModifyCoeffsForGuetzliDataStruct(cr420)
        cb = self.ModifyCoeffsForGuetzliDataStruct(cb420)

        # coeffs_DataFrame
        y_df = pd.DataFrame(y)
        cr_df = pd.DataFrame(cr)
        cb_df = pd.DataFrame(cb)

        # dump
        y_df.to_csv(self.csv_path + "y.csv", header=None, index=None)
        cr_df.to_csv(self.csv_path + "cr.csv", header=None, index=None)
        cb_df.to_csv(self.csv_path + "cb.csv", header=None, index=None)
    
    def create_generator_3layer(self):
        inputs = Input((self.target_size[0], self.target_size[1], 3))
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
        fcn = Model(input=inputs, output=conv8)

        return fcn

    def create_generator(self):
        # U-Net
        # EncoderSide
        inputs = Input((self.target_size[0], self.target_size[1], 3))

        conv1 = Conv2D(32, (3, 3), padding='same', data_format="channels_last")(inputs)
        conv1 = LeakyReLU(0.2)(conv1)

        conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv1)
        conv2 = LeakyReLU(0.2)(conv2)
        conv2 = BatchNormalization(axis=-1)(conv2)

        conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv2)
        conv3 = LeakyReLU(0.2)(conv3)
        conv3 = BatchNormalization(axis=-1)(conv3)

        conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv3)
        conv4 = LeakyReLU(0.2)(conv4)
        conv4 = BatchNormalization(axis=-1)(conv4)

        conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv4)
        conv5 = LeakyReLU(0.2)(conv5)
        conv5 = BatchNormalization(axis=-1)(conv5)

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
    
    @staticmethod
    def plot_heatmap(plotmap, path):
        plt.clf()
        plt.imshow(plotmap, cmap="hot", interpolation='nearest')
        plt.savefig(path)

    @staticmethod
    def resize444to420(coeffs):
        row = coeffs.shape[0]
        col = coeffs.shape[1]

        f_row = int(row / 2)
        f_col = int(col / 2)
        foundation = np.zeros((f_row, f_col))
        for j in range(0, row, 2):
            for i in range(0, col, 2):
                canvas4 = np.zeros((2, 2))
                for k in range(2):
                    for u in range(2):
                        canvas4[k][u] = coeffs[j+k][i+u]
                if np.sum(canvas4.flatten()) >= 2.0:
                    foundation[j//2][i//2] = 1
                else:
                    foundation[j//2][i//2] = 0
        return foundation

    @staticmethod
    def ModifyCoeffsForGuetzliDataStruct(coeffs):
        row = coeffs.shape[0]
        col = coeffs.shape[1]
        block_num = int((row * col) / 64)
        foundation = np.zeros((block_num, 64))
        block_ix = 0

        for j in range(0, row, 8):
            for i in range(0, col, 8):
                canvas64 = np.zeros((8, 8))
                for k in range(8):
                    for u in range(8):
                        canvas64[k][u] = coeffs[j+k][i+u]
                foundation[block_ix] = canvas64.flatten()
                block_ix += 1
        return foundation

    @staticmethod
    def change_threshold(predict_coeff, threshold):
        if predict_coeff >= threshold:
            return 1
        else:
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict dataset")
    parser.add_argument("--modelpath", "-m", type=str, 
                        required=True, help="load trained model weights")
    parser.add_argument("--imagepath", "-i", type=str, 
                        required=True, help="load Input Image")
    parser.add_argument("--targetsize", "-t", type=int,
                        default=224)
    parser.add_argument("--resultpath", "-r", type=str, default="test/heatmap/",
                        help="save output in this result path")
    parser.add_argument("--csvpath", "-c", type=str, default="test/coeffs_csv/",
                        help="default csv path")
    parser.add_argument("--guetzli_csv_path", "-gc", type=str, 
                        help="guetzli csv path")
    args = parser.parse_args()


    target_size = (args.targetsize, args.targetsize) # change to your image size
    # model_path, result_png_path, csv_path, guetzli_csv_path
    image = cv2.imread(args.imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) / 255.0
    predictor = Predictor(target_size, image, threshold=0.5,
                    model_path=args.modelpath, 
                    result_png_path=args.resultpath,
                    csv_path=args.csvpath,
                    guetzli_csv_path=args.guetzli_csv_path)
    predictor.predict()
    predictor.eval()
    predictor.plot(plot_guetzli_dct=True, binary=True)
    predictor.dump_csv()
