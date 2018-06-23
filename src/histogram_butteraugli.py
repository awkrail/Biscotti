import json
import numpy as np
import matplotlib.pyplot as plt

def main():
  validation_dirs = ["validations/biscotti_result224.json",
                    "validations/biscotti_result512.json",
                    "validations/biscotti_result1200.json"]
  
  # values histogram
  histo224= []
  histo512 = []
  histo1200 = []

  for validation_dir in validation_dirs:
    with open(validation_dir, "r") as f:
      data = json.load(f)
      for key, value in data.items():
        if value > 20:
          value = 21
        if validation_dir == "validations/biscotti_result224.json":
          histo224.append(value)
        elif validation_dir == "validations/biscotti_result512.json":
          histo512.append(value)
        else:
          histo1200.append(value)
  
  histos = [histo224, histo512, histo1200]

  plt.title("butteraugliのスコアの枚数")
  plt.ylabel("枚数")
  plt.xlabel("butteraugli")
  plt.hist([histos[0], histos[1], histos[2]], bins=21, color=['red', 'blue', 'green'], label=['224', '512', '1200'])
  plt.legend(loc="upper right")
  plt.show()
  # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
  main()