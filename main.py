import numpy as np

import pci as pci
import pandas as pd
from numpy import sin

from mathematics import dfop

# import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)


def x2(x): return x ** 2


# pci.PTest.generate_test(
#     function=sin,
#     in_set_initial_value=-100,
#     in_set_final_value=100,
#     in_set_step=0.5,
#     out_set_offset_range=[o for o in range(1, 15)],
#     out_set_rounder_range=[r for r in range(5, 30)],
#     force_train=True
# )

if __name__ == "__main__":
    # df = pd.read_csv("data\\output_sets\\generate\\sin_[-100,100]_0.5__o[(1, 14)]_r[(5, 29)].csv")
    # groups = dfop.split_by(df, ["offset", "rounder"])
    #
    # pci.PTest.plot_val_input_vs_error(groups[0], auto_upx=10, linestyle=":")

    # pci.PTest.generate_test(
    #     np.cos,
    #     -100,
    #     100,
    #     0.5,
    #     [o for o in range(1, 15)],
    #     [r for r in range(5, 30)],
    #     force_train=True
    #
    # )

    df = pd.read_csv("data\\output_sets\\generate\\cos_[-100,100]_0.5__o[(1, 14)]_r[(5, 29)].csv")
    groups = dfop.split_by(df, ["offset", "rounder"])

    for udf in groups:
        pci.PTest.plot_val_input_vs_error(udf, auto_upx=5, linestyle=":")
    pass
