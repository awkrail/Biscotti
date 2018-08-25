import cv2
import numpy as np
import os

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
    print("after  row col (", row+r_padding, ", ", col+c_padding, " )")
    return foundation

if __name__ == "__main__":
    image = cv2.imread("test/13.jpg")
    img_for_dct = change_image_size_to_dct(image)
    resized224_img = cv2.resize(img_for_dct, (224, 224))
    cv2.imwrite("test/img224_13.jpg", resized224_img)