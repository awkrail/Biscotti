import os
import argparse

def main(size, validation_dir):
  images = os.listdir(validation_dir)
  for image in images:
    # replace image
    ok = os.system("bin/Release/biscotti" validation_dir + "63893827.jpg " + validation_dir + "/opt_" + "63893827" + ".jpg")
    import ipdb; ipdb.set_trace()
  

if __name__ == "__main__":
  # 16で割り切れるという条件付き
  # TODO : この条件も本体で実装できたら消す
  parser = argparse.ArgumentParser(description="resize images and measure performance biscotti")
  parser.add_argument("--size", "-s", type=int,
                      required=True, help="image size")
  parser.add_argument("--valid_dir", "-v", type=str, help="validation directory")
  args = parser.parse_args()
  main(args.size, args.valid_dir)