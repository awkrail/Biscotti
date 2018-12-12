import os
import subprocess
import argparse
from datetime import datetime
import json

def main(validation_dir, save_dir):
  max_limit = 50 # change 1 ~ 1000
  image_files = sorted(os.listdir(validation_dir))[:max_limit]
  scores = []
  elapsed = []
  score_dict = {}

  for image_file in image_files:
    command = "bin/Release/biscotti"
    biscotti = [command, validation_dir + "/" + image_file, save_dir + "/" + image_file, 
                "pb_model/cropped_8.pb", "pb_model/cropped_8.pb"]
    try:
      result_dict = {
        "butteraugli" : -1,
        "file_size" : [],
        "elapsed_time" : -1
      }
      start = datetime.now()
      subprocess.check_call(biscotti)
      end = datetime.now()
      delta = end - start
      elapsed.append(delta.total_seconds())
      butteraugli = ["train_bin/Release/butteraugli", biscotti[1], biscotti[2]]
      score = subprocess.check_output(butteraugli)
      score = float(score)
      print("score : ", score)
      print("delta : ", delta)
      scores.append(score)
      result_dict["butteraugli"] = score
      result_dict["elapsed_time"] = delta.total_seconds()
      before_size = os.path.getsize(biscotti[1]) / 1000
      after_size = os.path.getsize(biscotti[2]) / 1000
      result_dict["file_size"].append(before_size)
      result_dict["file_size"].append(after_size)
      score_dict[image_file] = result_dict
    except:
      print("error!")

  print(" --- stats --- ")
  print("the number of images : ", len(image_files))
  print("the number of processed images : ", len(scores))
  print("minimum butteraugli : ", min(scores))
  print("maximum butteraugli : ", max(scores))
  print("average butteraugli : ", sum(scores) / len(scores)) 
  print("average elapsed time :", sum(elapsed) / len(elapsed))

  """
  結果を出力
  """
  json_dir = "safari_results/biscotti_result.json"
  with open(json_dir, "w") as f:
    data = json.dumps(score_dict, f)
    f.write(data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="resize images and measure performance biscotti or guetzli")
  parser.add_argument("--valid_dir", "-v", type=str,
                      required=True, help="validation directory")
  parser.add_argument("--save_dir", "-s", type=str,
                      required=True, help="save dir")
  args = parser.parse_args()
  main(args.valid_dir, args.save_dir)