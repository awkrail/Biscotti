import json
import numpy as np
import matplotlib.pyplot as plt

"""
validations/ 以下の結果から
  1. butteraugli
  2. 経過時間(elapsed time) : こいつは平均でいいような気がする
  3. ファイルサイズ : beforeを横軸, afterを縦軸にグラフをplotする

"""

def check_image_size(validation_dir):
  if validation_dir.find("224") > 0:
    return 224
  elif validation_dir.find("512") > 0:
    return 512
  elif validation_dir.find("1200") > 0:
    return 1200
  else:
    return -1 # Error
  
def size_array_to_x_and_y(size_array):
  x = np.array([value[0] for value in size_array], dtype=np.float32)
  y = np.array([value[1] for value in size_array], dtype=np.float32)
  return x, y

def print_result(title, array224, array512, array1200):
  print("========== ", title, " ==========")
  print("size 224 : ", "min : ", min(array224), " max : ", max(array224), " average : ", sum(array224) / len(array224))
  print("size 512 : ", "min : ", min(array512), " max : ", max(array512), " average : ", sum(array512) / len(array512))
  print("size 1200 : ", "min : ", min(array1200), " max : ", max(array1200), " average : ", sum(array1200) / len(array1200))
  print("=================================")
  print("")

def main():
  # butteraulgi score
  validation_dirs = ["validations/results420/biscotti_result224.json",
                    "validations/results420/biscotti_result512.json",
                    "validations/results420/biscotti_result1200.json"
                    ]
  
  # butteraugli histogram
  histo224= []
  histo512 = []
  histo1200 = []

  # butteraugli value
  butteraugli224 = []
  butteraugli512 = []
  butteraugli1200 = []

  # filesize plot
  size224 = []
  size512 = []
  size1200 = []

  # elapsed time
  elapsed_time_224 = []
  elapsed_time_512 = []
  elapsed_time_1200 = []

  for validation_dir in validation_dirs:
    size = check_image_size(validation_dir)
    with open(validation_dir, "r") as f:
      data = json.load(f)
      for key, value in data.items():
        butteraugli = value["butteraugli"]
        before_size = value["file_size"][0]
        after_size = value["file_size"][1]
        elapsed_time = value["elapsed_time"]
        if butteraugli > 10:
          butteraugli = 11
        if size == 224:
          histo224.append(int(butteraugli))
          butteraugli224.append(butteraugli)
          size224.append([before_size, after_size])
          elapsed_time_224.append(elapsed_time)
        elif size == 512:
          histo512.append(int(butteraugli))
          butteraugli512.append(butteraugli)
          size512.append([before_size, after_size])
          elapsed_time_512.append(elapsed_time)
        elif size == 1200:
          histo1200.append(int(butteraugli))
          butteraugli1200.append(butteraugli)
          size1200.append([before_size, after_size])
          elapsed_time_1200.append(elapsed_time)
        else:
          print("[Error] : no such file size")
          exit(1)
  
  histos = [histo224, histo512, histo1200]

  plt.title("butteraugliのスコアの枚数")
  plt.ylabel("枚数")
  plt.xlabel("butteraugli")
  # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  # ここももっと詳細にbutteraugliのスコアのヒストグラムがわかるようにするべきでは?
  plt.hist([butteraugli224, butteraugli512, butteraugli1200], align='mid',
    color=['red', 'blue', 'green'], label=['224', '512', '1200'])
  # plt.hist(butteraugli224)
  plt.legend(loc="upper right")
  plt.savefig("results/biscotti_histogram_fixed.png")

  # file_size
  plt.clf()
  plt.title("file sizeの変化(KB)")
  plt.ylabel("after")
  plt.xlabel("before")
  x, y = size_array_to_x_and_y(size224)
  plt.scatter(x, y, color="red", marker="o", alpha=0.3, label="224")
  x, y = size_array_to_x_and_y(size512)
  plt.scatter(x, y, color="blue", marker="o", alpha=0.3, label="512")
  x, y = size_array_to_x_and_y(size1200)
  plt.scatter(x, y, color="green", marker="o", alpha=0.3, label="1200")
  x_1, y_1 = np.arange(1000), np.arange(1000)
  plt.plot(x_1, y_1, color="m", label="")
  y_1_2 = y_1 * (1/2)
  plt.plot(x_1, y_1_2, color="c")
  plt.legend(loc="upper right")
  plt.savefig("results/change_file_size.png")

  # print elapsed time
  size224 = [value[1] for value in size224]
  size512 = [value[1] for value in size512]
  size1200 = [value[1] for value in size1200]
  print_result("elapsed_time", elapsed_time_224, elapsed_time_512, elapsed_time_1200)
  print_result("butteraugli", butteraugli224, butteraugli512, butteraugli1200)
  print_result("filesize", size224, size512, size1200)


if __name__ == "__main__":
  main()