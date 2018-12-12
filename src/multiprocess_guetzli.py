import argparse
from multiprocessing import Process
import subprocess
import os

def guetzli_dumper(input_file_path, input_q_file_path, input_o_file_path):
  command = ["train_bin/Release/guetzli_dumper", input_file_path, input_q_file_path, input_o_file_path]
  try:
    subprocess.check_call(command)
  except:
    pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="multiprocess guetzli's dumping csv")
  parser.add_argument("--input_file_path", "-i", type=str, required=True)
  parser.add_argument("--output_file_path", "-o", type=str, required=True)
  args = parser.parse_args()
  input_files = os.listdir(args.input_file_path)
  
  # 10個ずつ処理
  for i in range(0, len(input_files), 10):
    batch_input_files = input_files[i:i+10]
    batch_input_file_path = ["aug_cropped_images/" + input_file for input_file in input_files]
    batch_q_output_file_path = ["aug_cropped_q_images/" + input_file for input_file in input_files]
    batch_o_output_file_path = ["aug_cropped_o_images/" + input_file for input_file in input_files]
    # p = 
    p = Process(target=guetzli_dumper, args=(batch_input_files, batch_q_output_file_path, batch_o_output_file_path))
    p.start()
    print("start process")
    p.join()
