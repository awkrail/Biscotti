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

        WATCH: After guetzli, images is transformed into YUV422, so you can think label data Y:Cb:Cr = 2:1:1
        """
        pass
    
    def dct_csv2numpy_probability(self):
        checker_0 = np.vectorize(self.check0)
        for csv_file in os.listdir(self.csv_path):
            csv_numpy = pd.read_csv(self.csv_path + "/" + csv_file, header=None).get_values()
            binary = checker_0(csv_numpy == 0)
            yield binary
    
    def make_images_and_labels(self):
        qopt_files = os.listdir(self.qopt_path)
        # images and labels
        # TODO: imread => YCbCrで保存しておく
        images = [cv2.imread(self.qopt_path + "/" + q_file) for q_file in qopt_files]
        labels = self.dct_csv2numpy_probability()

        # TODO: coeff_y, coeff_cb, coeff_crを画像と同じ形にする
        for img, label in zip(images, labels):
            seq = int(label.shape[0]/2) 
            coeff_y, coeff_cbcr = label[:seq], label[seq:]
            coeff_cb = np.array(list(map(self.resize422to444, coeff_cbcr[:int(seq/2)].tolist())))
            coeff_cr = np.array(list(map(self.resize422to444, coeff_cbcr[int(seq/2):].tolist())))
            import ipdb; ipdb.set_trace()




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