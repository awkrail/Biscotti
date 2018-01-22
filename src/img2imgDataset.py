import numpy as np
import cv2
import os
import pandas as pd

"""
[WIP]

this class operates the dataset.
except for load_dataset() method, other method is used for making dataset.

csv/ => dumped DCT coeffs.
qopt_images/ => images made by guetzli, and this image is only optimized quantization table, but not DCT coeffs.
labels/ => read csv(DCT coeffs), and if each coeff is not 0, set a value 1. if each coeff is 0, set a value 0.

"""


class Image2ImageDataset(object):
    def __init__(self):
        self.qopt_path = "qopt_images/"
        self.label_path = "label/"
        self.csv_path = "csv/"
    
    def load_dataset(self):
        """
        this function is used for training scripts.
        other functions is used for making dataset.

        WATCH: After guetzli, images is transformed into YUV420, so you can think label data Y:Cb:Cr = 4:1:1
        """
        pass
    
    def dct_csv2numpy_probability(self):
        checker_0 = np.vectorize(self.check0)
        for csv_file in os.listdir(self.csv_path):
            if csv_file.startswith("."):
                continue
            csv_numpy = pd.read_csv(self.csv_path + "/" + csv_file, header=None).get_values()
            yield checker_0(csv_numpy == 0)
    
    def make_images_and_labels(self):
        qopt_files = os.listdir(self.qopt_path)
        # images and labels
        images = [cv2.imread(self.qopt_path + "/" + q_file) for q_file in qopt_files if q_file != ".DS_Store"]
        labels = self.dct_csv2numpy_probability()

        # TODO: coeff_y, coeff_cb, coeff_crを画像と同じ形にする
        # TODO: GuetzliはYUV444も採用する可能性があるので次元数とshapeを確認する。
        for img, label in zip(images, labels):
            height = img.shape[0]
            width = img.shape[1]

            height_blocks = int(height / 8)
            width_blocks = int(width / 8)

            seq = int(label.shape[0] * (2/3))
            coeff_y, coeff_cbcr = label[:seq], label[seq:]
            coeff_y = self.resize_coeff_to_img_matrix(coeff_y, width, height)
            coeff_cb = coeff_cbcr[:int(seq/4)]
            coeff_cr = coeff_cbcr[int(seq/4):]
            coeff_cb = self.resize420to444(coeff_cb, width, height)
            coeff_cr = self.resize420to444(coeff_cr, width, height)



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
        return canvas

    @staticmethod
    def check0(coeff):
        if not coeff:
            return 1
        else:
            return 0
    
    # TODO: 間引きさせる関数の作成
    @staticmethod
    def resize420to444(coeff, width, height):
        canvas = np.zeros((height, width))
        width_blocks = int(width / 8)
        height_blocks = int(height / 8)
        for block_y in range(width_blocks):
            for block_x in range(height_blocks):
                canvas16 = np.zeros((16, 16)).astype(np.int32)
                block_ix = height_blocks * block_y + block_x
                block = coeff[block_ix].reshape(8, 8)
                for i in range(8):
                    for j in range(8):
                        dct22 = block[i][j] * np.ones((2, 2)).astype(np.int32)
                        canvas16[j*2:j*2+2, i*2:i*2+2] = dct22
                canvas[block_x*16:block_x*16+16, block_y*16:block_y*16+16] = canvas16
        return canvas


if __name__ == "__main__":
    dataset = Image2ImageDataset()
    dataset.make_images_and_labels()