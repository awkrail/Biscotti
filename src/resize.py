import cv2
import numpy as np
import os

"""
this script changes images size to the DCT table size.
"""


def change_image_size_to_dct(image):
    row = image.shape[0]
    col = image.shape[1]

    mod_r_8 = row % 8
    mod_c_8 = col % 8

    r_padding = 0
    c_padding = 0

    if mod_r_8 != 0:
        r_padding += (8 - mod_r_8)
        
    if mod_c_8 != 0:
        c_padding += (8 - mod_c_8)
        
    foundation = np.zeros((row+r_padding, col+c_padding, 3))
    foundation[:image.shape[0], :image.shape[1], :image.shape[2]] = image
    print("before row, col (", row, ", ", col, " )")
    print("after  row col (", row+r_padding, ", ", col, " )")
    return foundation

if __name__ == "__main__":
    print("=== resizing dataset ... ===")
    images_files = os.listdir("images/")
    images = [cv2.imread("images/" + image_file) for image_file in images_files]
    
    for pad_image, image_file in zip(map(change_image_size_to_dct, images), images_files):
        print(image_file ," done!")
        cv2.imwrite("resized_images/" + image_file, pad_image)
    print("===> Done!")