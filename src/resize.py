import cv2
import numpy as np
import os

"""
this script changes images size to the DCT table size.
"""


class ImageRisizer(object):
    def __init__(self, image_files):
        self.image_files = sorted(image_files)
    
    def resize(self, predict=False):
        images = self.yield_image()
        for i, image in enumerate(images):
            if image is None:
                print("may be broken file")
                continue
            if predict:
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
                
                new_row = row + mod_r_8
                new_col = col + mod_c_8
                resized_image = cv2.resize(image, (new_row, new_col), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite("resized_images/" + str(i) + ".jpg", resized_image)
            else:
                resized_image = cv2.resize(image, (224, 224))
                cv2.imwrite("resized_images/" + str(i) + ".jpg", resized_image)
                print(i, "done!")
    
    def yield_image(self):
        for image_file in self.image_files:
            yield cv2.imread("images/" + image_file)

if __name__ == "__main__":
    print("=== resizing dataset ... ===")
    image_files = os.listdir("images/")
    resizer = ImageRisizer(image_files)
    resizer.resize()
    print("===> Done!")
