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

        WATCH: After guetzli, images is transformed into YUV420, so you can think label data Y:Cb:Cr = 2:1:1
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
        # TODO: imread => YCbCrで保存しておく
        images = [cv2.imread(self.qopt_path + "/" + q_file) for q_file in qopt_files if q_file != ".DS_Store"]
        labels = self.dct_csv2numpy_probability()

        # TODO: coeff_y, coeff_cb, coeff_crを画像と同じ形にする
        for img, label in zip(images, labels):
            height = img.shape[0]
            width = img.shape[1]

            height_blocks = int(height / 8)
            width_blocks = int(width / 8)

            seq = int(label.shape[0] * (2/3))
            coeff_y, coeff_cbcr = label[:seq], label[seq:]
            coeff_y = self.resize_coeff_to_img_matrix(coeff_y, width, height)
            import ipdb; ipdb.set_trace()
            coeff_cb = coeff_cbcr[:int(seq/4)]
            coeff_cr = coeff_cbcr[int(seq/4):]
    
    @staticmethod
    def resize_coeff_to_img_matrix(coeff, width, height):
        canvas = np.zeros((height, width))
        width_blocks = int(width / 8)
        height_blocks = int(height / 8)
        num_blocks = width_blocks * height_blocks
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
    
    @staticmethod
    def resize422to444(coeff_c):
        return [coeff_c, coeff_c]


if __name__ == "__main__":
    dataset = Image2ImageDataset()
    dataset.make_images_and_labels()