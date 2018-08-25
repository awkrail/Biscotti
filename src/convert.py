import numpy as np
import pandas as pd
import argparse

def main():
    guetzli = pd.read_csv('csv/0.csv', header=None).get_values()
    seq = int(guetzli.shape[0] * (2/3))
    coeff_y, coeff_cbcr = guetzli[:seq], guetzli[seq:]
    coeff_cb = coeff_cbcr[:int(seq/4)]
    coeff_cr = coeff_cbcr[int(seq/4):]

    y_df = pd.DataFrame(coeff_y)
    cb_df = pd.DataFrame(coeff_cb)
    cr_df = pd.DataFrame(coeff_cr)

    y_df.to_csv("csv/0_biscotti_y.csv", header=None, index=None)
    cb_df.to_csv("csv/0_biscotti_cb.csv", header=None, index=None)
    cr_df.to_csv("csv/0_biscotti_cr.csv", header=None, index=None)

if __name__ == "__main__":
    main()