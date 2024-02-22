import pci as pci
import pandas as pd
from numpy import sin

# import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)


def x2(x): return x ** 2

#
# pci.PTest.generate_test(
#     function=x2,
#     in_set_initial_value=100,
#     in_set_final_value=200,
#     in_set_step=2,
#     out_set_offset_range=[o for o in range(1, 10)],
#     out_set_rounder_range=[r for r in range(5, 15)],
#     force_train=True
# )

# a = pci.PTest.uniform_data_range(
#     pd.read_csv("data\\input_sets\\generate\\x2_[1-10]_2.csv"),
#     x2,
#     [o for o in range(1, 6)],
#     [r for r in range(10, 15)],
#     force_train=True
# )


# if __name__ == "__main__":
#
#     lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
#     for i, val in enumerate(lst):
#
#         print(i)
#         print(val)
#
#
#     pass
