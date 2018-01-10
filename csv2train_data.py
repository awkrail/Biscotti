import numpy as np 
import os
import pandas as pd

def check0(coeff):
    if not coeff:
        return 1
    else:
        return 0


class TrainDataMarker(object):
    def __init__(self):
        self.csv_files = os.listdir("csv/")
    
    def dct2numpy_prob(self):
        check_0 = np.vectorize(check0)
        for csv_file in self.csv_files:
            csv_np = pd.read_csv("csv/" + csv_file, header=None).get_values()
            binary_np = check_0(csv_np == 0)
            np.save("train/" + csv_file[:-4], binary_np)
            print("file" + csv_file + " done!")

if __name__ == "__main__":
    td_maker = TrainDataMarker()
    td_maker.dct2numpy_prob()