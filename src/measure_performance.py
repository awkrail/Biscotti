import os
import subprocess
import argparse
from datetime import datetime
import json

def main(validation_dir, save_dir, is_guetzli):
  images = os.listdir(validation_dir)
  grayscale = 0
  scores = []
  elapsed = []
  score_dict = {}
  print(len(images))
  if is_guetzli:
    command = "guetzli"
  else:
    command = "bin/Release/biscotti"
  for image in images:
    # TODO : 16で両辺が割り切れないとダメ => それへの対応
    # TODO : 1000枚に実行するので遅い。夜に実行する
    # replace image
    biscotti = [command, validation_dir + "/" + image, save_dir + "/" + image, "pb_model/output_graph_360.pb"]
    try:
      start = datetime.now()
      subprocess.check_call(biscotti)
      end = datetime.now()
      delta = end - start
      elapsed.append(delta.total_seconds())
      butteraugli = ["train_bin/Release/butteraugli", biscotti[1], biscotti[2]]
      score = subprocess.check_output(butteraugli)
      score = float(score)
      print(score)
      scores.append(score)
      score_dict[image] = score
    except:
      grayscale += 1

  print(" --- stats --- ")
  print("the number of images : ", len(images))
  print("except for grayscale : ", len(scores))
  print("grayscale : ", grayscale)
  print("minimum butteraugli : ", min(scores))
  print("maximum butteraugli : ", max(scores))
  print("average butteraugli : ", sum(scores) / len(scores)) 
  print("average elapsed time :", sum(elapsed) / len(elapsed))
  # import ipdb; ipdb.set_trace()
  if save_dir[-3:] == "512":
    json_dir = "validations/result512.json"
  elif save_dir[-3:] == "224":
    json_dir = "validations/result224.json"
  else:
    json_dir = "validations/result1200.json"
  with open(json_dir, "w") as f:
    json.dump(score_dict, f)


if __name__ == "__main__":
  # 16で割り切れるという条件付き
  # TODO : この条件も本体で実装できたら消す
  parser = argparse.ArgumentParser(description="resize images and measure performance biscotti or guetzli")
  parser.add_argument("--valid_dir", "-v", type=str,
                      default="validations/images/", help="validation directory")
  parser.add_argument("--save_dir", "-s", type=str,
                      required=True, help="save dir")
  parser.add_argument("--guetzli", "-g", type=bool,
                      default=False, help="measure performance guetzli or biscotti. if you want to compress images with guetzli, please add -g True")
  args = parser.parse_args()
  main(args.valid_dir, args.save_dir, args.guetzli)