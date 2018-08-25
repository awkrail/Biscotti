import argparse
import cv2
import os

def main(validation_dir, size):
  assert(size == 224 or size == 512 or size == 1200)
  images = os.listdir(validation_dir)
  for image in images:
    print(image)
    cv_img = cv2.imread(validation_dir + image)
    if cv_img is None:
      os.system("rm " + validation_dir + image)
      continue
    resized_img = cv2.resize(cv_img, (size, size))
    if size == 224:
      save_dir = "validations/images224/"
    elif size == 512:
      save_dir = "validations/images512/"
    else:
      save_dir = "validations/images1200/"
    cv2.imwrite(save_dir + image, resized_img)
    print(image, ", done!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="resize images")
  parser.add_argument("--size", "-s", type=int,
                      required=True, help="resize image size")
  parser.add_argument("--validation_dir", "-v", type=str,
                      default="validations/images1200/", help="validation image directory")
  args = parser.parse_args()
  main(args.validation_dir, args.size)