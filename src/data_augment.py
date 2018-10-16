import cv2
import numpy as np
from scipy.misc import imresize
import argparse

def random_crop(image, crop_size=(224, 224)):
  h, w, _ = image.shape

  # (0 ~ 224)の間でtop, left
  top = np.random.randint(0, h - crop_size[0])
  left = np.random.randint(0, w - crop_size[1])

  # bottom, rightを決める
  bottom = top + crop_size[0]
  right = left + crop_size[1]

  # 決めたtop, bottom, left, rightを使って画像を抜き出す
  image = image[top:bottom, left:right, :]
  return image

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Random Crop for guetzli")
  parser.add_argument("--input_images", "-i", type=str, required=True)
  parser.add_argument("--output_images", "-o", type=str, required=True)
  args = parser.parse_args()

  image_files = os.listdir(args.input_images)
  for image_file in image_files:
    image = cv2.imread(args.input_images + image_file)
    for i in range(10):
      cropped_image = random_crop(image)
      cv2.imwrite(args.output_images + image_file + "_" + str(i) + ".jpg")
