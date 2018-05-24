import os
import argparse

def main():
  images = os.listdir("validations/images/")
  for image in images:
    pass
  

if __name__ == "__main__":
  # 16で割り切れるという条件付き
  # TODO : この条件も本体で実装できたら消す
  # TODO : GrayScaleはデータとして受け付けないように処理を変える
  parser = argparse.ArgumentParser(description="resize images and measure performance biscotti")
  main()