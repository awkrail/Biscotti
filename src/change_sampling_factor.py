# To make dataset, change sampling factor with Imagemagick.
import argparse
import os
import subprocess

def main(args):
  input_dir = args.input_images
  sampling_factor = args.sampling_factor
  output_dir = args.output_images

  for i_path in os.listdir(input_dir):
    if sampling_factor == 420:
      # command = ["convert", input_dir + i_path, '-sampling-factor "2x2,1x1,1x1"', output_dir + i_path]
      command = "convert " + input_dir + i_path + " -sampling-factor 4:2:0 " + output_dir + i_path
    else:
      command = "convert " + input_dir + i_path + " -sampling-factor 4:4:4 " + output_dir + i_path
    subprocess.call(command, shell=True)
    print(i_path + " done!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Making dataset for training biscotti")
  parser.add_argument("--input_images", '-i', type=str, required=True) # path of input images
  parser.add_argument("--sampling_factor", '-s', type=int, required=True) # 420 or 444
  parser.add_argument("--output_images", '-o', type=str, required=True) # path of output images
  args = parser.parse_args()
  main(args)
  print("Done!")