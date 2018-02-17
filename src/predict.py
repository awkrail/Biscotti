import numpy as np
import pandas as pd
import cv2
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
# from keras.layers import merge
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import list_pictures, array_to_img
import matplotlib.pyplot as plt


class DctCsvLoader(object):
    def __init__(self, guetzli_csv_path):
        self.csv_path = guetzli_csv_path
        self.dct = pd.read_csv(guetzli_csv_path, header=None).get_values()
    
    def get_csv(self):
        label = self.dct_csv2numpy_probability()
        seq = int(label.shape[0] * (2/3))
        width = 224
        height = 224

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
                block_ix = height_blocks * block_y + block_x
                block = coeff[block_ix].reshape(8, 8)
                canvas[block_x*8:block_x*8+8, block_y*8:block_y*8+8] = block
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
                block_ix = height_blocks * block_y + block_x
                block = coeff[block_ix].reshape(8, 8)
                for i in range(8):
                    for j in range(8):
                        dct22 = block[i][j] * np.ones((2, 2)).astype(np.int32)
                        canvas16[j*2:j*2+2, i*2:i*2+2] = dct22
                canvas[block_x*16:block_x*16+16, block_y*16:block_y*16+16] = canvas16
        return canvas.reshape(height, width, 1)


class Predictor():
    def __init__(self, target_size, image, model_path, result_png_path, csv_path, guetzli_csv_path):
        self.target_size = target_size
        self.image = image
        self.result_png_path = result_png_path
        self.csv_path = csv_path
        self.model = self.create_fcn()
        self.model.load_weights(model_path)
        self.guetzli_dct = DctCsvLoader(guetzli_csv_path).get_csv()
        self.predict_dct = None
    
    def predict(self):
        dct_predict = self.model.predict(image.reshape(1, 224, 224, 3)).reshape(224, 224, 3)
        self.predict_dct = dct_predict

    def eval(self):
        # Y accuracy
        dct_binary = np.round(self.predict_dct)
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
            dct_binary = np.round(self.predict_dct)
            for i in range(3):
                self.plot_heatmap(self.guetzli_dct[:, :, i], self.result_png_path + paths[i])
        for i in range(3):
            self.plot_heatmap(self.predict_dct[:, :, i], self.result_png_path + paths[i])
    
    def dump_csv(self):
        dct_binary = np.round(self.predict_dct)
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
    
    def create_fcn(self):
        inputs = Input((self.target_size[0], self.target_size[1], 3))
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
        pool5 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv5) # 元は(2, 2)

        conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool5)
        conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv6)

        # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
        up7 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv5], axis=3) # 元は(2, 2)
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


if __name__ == "__main__":
    target_size = (224, 224) # change to your image size
    image = cv2.imread("test/img224.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) / 255.0
    predictor = Predictor(target_size, image, "checkpoints/model_weights_150.h5", "test/heatmap/", "test/coeffs_csv/", "test/guetzli_img224.csv")
    predictor.predict()
    predictor.eval()
    predictor.plot()
    predictor.dump_csv()