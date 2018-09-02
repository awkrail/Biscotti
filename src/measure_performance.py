import os
import subprocess
import argparse
from datetime import datetime
import json

def check_file_size(validation_dir):
  if validation_dir.find("224") > 0:
    return 224
  elif validation_dir.find("512") > 0:
    return 512
  elif validation_dir.find("1200") > 0:
    return 1200
  else:
    return -1

def main(validation_dir, save_dir, is_guetzli, sampling):
  images = os.listdir(validation_dir)
  yuv444 = 0
  scores = []
  elapsed = []
  score_dict = {}
  print(len(images))
  if is_guetzli:
    command = "guetzli"
  else:
    command = "bin/Release/biscotti"
  for image in images:
    biscotti = [command, validation_dir + "/" + image, save_dir + "/" + image, 
                "pb_model/output_graph_360.pb", "pb_model/output_model_444.pb"]
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
      print(score)
      scores.append(score)
      result_dict["butteraugli"] = score
      result_dict["elapsed_time"] = delta.total_seconds()
      before_size = os.path.getsize(biscotti[1]) / 1000
      after_size = os.path.getsize(biscotti[2]) / 1000
      result_dict["file_size"].append(before_size)
      result_dict["file_size"].append(after_size)
      score_dict[image] = result_dict
    except:
      print("error")

  print(" --- stats --- ")
  print("the number of images : ", len(images))
  print("except for grayscale : ", len(scores))
  print("yuv444 : ", yuv444)
  print("minimum butteraugli : ", min(scores))
  print("maximum butteraugli : ", max(scores))
  print("average butteraugli : ", sum(scores) / len(scores)) 
  print("average elapsed time :", sum(elapsed) / len(elapsed))

  filesize = check_file_size(validation_dir)
  
  if sampling == 420:
    sampling_dir = "results420"
  else:
    sampling_dir = "results444"

  if filesize == 224:
    json_dir = "validations/" + sampling_dir + "/biscotti_result224.json"
  elif filesize == 512:
    json_dir = "validations/" + sampling_dir + "/biscotti_result512.json"
  elif filesize == 1200:
    json_dir = "validations/" + sampling_dir + "/biscotti_result1200.json"
  else:
    print("[Error] : file size must be 224, 512, or 1200")
    exit(1)

  with open(json_dir, "w") as f:
    data = json.dumps(score_dict, f)
    f.write(data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="resize images and measure performance biscotti or guetzli")
  parser.add_argument("--valid_dir", "-v", type=str,
                      default="validations/images/", help="validation directory")
  parser.add_argument("--save_dir", "-s", type=str,
                      required=True, help="save dir")
  parser.add_argument("--guetzli", "-g", type=bool,
                      default=False, help="measure performance guetzli or biscotti. if you want to compress images with guetzli, please add -g True")
  parser.add_argument("--sampling", "-samp", type=int,
                      default=420, help="chroma sub sampling, 420 or 444")
  args = parser.parse_args()
  main(args.valid_dir, args.save_dir, args.guetzli, args.sampling)