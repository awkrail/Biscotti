import os
import subprocess
import argparse

def main(size, validation_dir):
  images = os.listdir(validation_dir)
  grayscale = 0
  scores = []
  for image in images:
    # TODO : 16で両辺が割り切れないとダメ => それへの対応
    # TODO : 1000枚に実行するので遅い。夜に実行する
    # replace image
    biscotti = ["bin/Release/biscotti", validation_dir + image, validation_dir + "/opt_" + image]
    try:
      subprocess.check_call(biscotti)
      butteraugli = ["bin/Release/butteraugli", biscotti[1], biscotti[2]]
      score = subprocess.check_output(butteraugli)
      score = float(score)
      scores.append(score)
    except:
      grayscale = 0

  print(" --- stats --- ")
  print("the number of images : ", len(images))
  print("except for grayscale : ", len(scores))
  print("grayscale : ", grayscale)
  print("minimum butteraugli : ", min(scores))
  print("maximum butteraugli : ", max(scores))
  print("average butteraugli : ", sum(scores) / len(scores)) 

if __name__ == "__main__":
  # 16で割り切れるという条件付き
  # TODO : この条件も本体で実装できたら消す
  parser = argparse.ArgumentParser(description="resize images and measure performance biscotti")
  parser.add_argument("--size", "-s", type=int,
                      required=True, help="image size")
  parser.add_argument("--valid_dir", "-v", type=str,
                      default="validations/images/", help="validation directory")
  args = parser.parse_args()
  main(args.size, args.valid_dir)