import cv2
import numpy as np
import os
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
    file = image_file.split(".")[0]
    image = cv2.imread(args.input_images + image_file)
    if image is None:
      continue
    for i in range(10):
      cropped_image = random_crop(image)
      cv2.imwrite(args.output_images + file + "_" + str(i) + ".jpg", cropped_image)
      print(args.output_images + file + "_" + str(i) + ".jpg" + ", done!")
    print(image_file + ", done!")
