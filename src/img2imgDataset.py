import numpy as np
import cv2
import os
import pandas as pd
import argparse

"""
[WIP]

this class operates the dataset.
except for load_dataset() method, other method is used for making dataset.

csv/ => dumped DCT coeffs.
qopt_images/ => images made by guetzli, and this image is only optimized quantization table, but not DCT coeffs.
labels/ => read csv(DCT coeffs), and if each coeff is not 0, set a value 1. if each coeff is 0, set a value 0.


WATCH: After guetzli, images is transformed into YUV420 or YUV444, so you can think label data Y:Cb:Cr = 4:1:1
TODO: this script doesn't accept YUV444 images.
TODO: this script doesn't accept gray scale.
"""


class Image2ImageDataset(object):
    def __init__(self, qopt_images_path, train_path, csv_path, sampling_factor):
        self.qopt_path = qopt_images_path
        self.train_path = train_path
        self.csv_path = csv_path
    
    def load_yield_image(self, qopt_files):
        for qopt_file in qopt_files:
            if qopt_file.startswith("."):
                continue
            yield cv2.imread(self.qopt_path + "/" + qopt_file)

    def dct_csv2numpy_probability(self, csv_files):
        checker_0 = np.vectorize(self.check0)
        for csv_file in csv_files:
            if csv_file.startswith("."):
                continue
            csv_numpy = pd.read_csv(self.csv_path + "/" + csv_file, header=None).get_values()
            yield checker_0(csv_numpy == 0)
    
    def make_images_and_labels(self):
        """
        this method make dataset.
        now you can only make 3d(YCrCb)Dataset, so you should exclude gray scale images.
        """
        # file name should be same qopt_files and csv_files.
        # In my case, file name is used increment number.

        qopt_files = sorted(os.listdir(self.qopt_path))
        csv_files = sorted(os.listdir(self.csv_path))

        if not self.assert_two_lists_is_same(qopt_files, csv_files):
            print("Please check your qopt_images/ and csv/ are same.")
            exit()

        images = [cv2.imread(self.qopt_path + "/" + q_file) for q_file in qopt_files if not q_file.startswith(".")]
        labels = self.dct_csv2numpy_probability(csv_files)
        for i in range(len(qopt_files)):
            img = images[i]
            label = next(labels)
            filename = qopt_files[i].replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            print(filename)

            if self.check_grayscale(img, label): # grayscaleではじくのいらん気がする
                print("this image is on gray scale data!")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) / 255.0
            height = img.shape[0]
            width = img.shape[1]
            if self.check_chroma_subsampling(img, label):
                # YUV444
                seq = int(label.shape[0] * (1/3))
                coeff_y, coeff_cb, coeff_cr = label[:seq], label[seq:2*seq], label[2*seq:] # 勘で書いてるのでチェックする
                coeff_y = self.resize_coeff_to_img_matrix(coeff_y, width, height)
                coeff_cb = self.resize_coeff_to_img_matrix(coeff_cb, width, height)
                coeff_cr = self.resize_coeff_to_img_matrix(coeff_cb, width, height)
                coeff3d = np.concatenate((coeff_y, coeff_cr, coeff_cb), axis=2)
                result = np.concatenate((img, coeff3d), axis=2)
                np.save(self.train_path + filename + ".npy", result)
                print(filename, "has done!")
            else:
                # YUV420
                height = img.shape[0]
                width = img.shape[1]
                seq = int(label.shape[0] * (2/3))
                coeff_y, coeff_cbcr = label[:seq], label[seq:]
                coeff_y = self.resize_coeff_to_img_matrix(coeff_y, width, height)
                coeff_cb = self.resize420to444(coeff_cbcr[:int(seq/4)], width, height)
                coeff_cr = self.resize420to444(coeff_cbcr[int(seq/4):], width, height)
                coeff3d = np.concatenate((coeff_y, coeff_cr, coeff_cb), axis=2)
                result = np.concatenate((img, coeff3d), axis=2)
                np.save(self.train_path + filename + ".npy", result)
                print(filename, "has done!")

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

    @staticmethod
    def check_grayscale(image, label):
        return True if image.shape[0]*image.shape[1] == label.shape[0]*label.shape[1] else False
    
    @staticmethod
    def check_chroma_subsampling(image, label):
        return True if image.shape[0] * image.shape[1] * image.shape[2] == label.shape[0] * label.shape[1] else False
    
    @staticmethod
    def assert_two_lists_is_same(jpg_files, csv_files):
        for jpg_file, csv_file in zip(jpg_files, csv_files):
            if jpg_file.replace(".jpg", "") != csv_file.replace(".csv", ""):
                return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making dataset for training biscotti")
    parser.add_argument("--qopt_images", '-q', type=str, default="qopt_images/")
    parser.add_argument("--csv", "-c", type=str, default="csv/")
    parser.add_argument("--train_path", "-t", type=str, default="train/")
    args = parser.parse_args()

    qopt_images_path = args.qopt_images
    csv_path = args.csv
    train_path = args.train_path

    print("=== making dataset... ===")
    dataset = Image2ImageDataset(qopt_images_path, train_path, csv_path)
    dataset.make_images_and_labels()
    print("===> Done!")
