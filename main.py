# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:26:47 2023

@author: alvar
"""

from pci import PCI
import error as err
from pandas import read_csv
from pandas import DataFrame, Series

from numpy import arange, array, sin

import matplotlib.pyplot as plt

def cuac(x):
    return x**2

df : DataFrame = read_csv("C:\\Users\\ZiftK\\Desktop\\TODO\\Python\\pci\\csvs\\sin2.csv")

#df = DataFrame({"x":arange(1,6),"y":[x**2 for x in range(1,6)]})
pcys = PCI(df,30)

val_range = arange(1,9,0.01)
graph_range = arange(1,9,0.5)

edf = err.error_to_plot(pcys, sin,val_range)

print(pcys)

plt.plot(edf)
plt.xticks(graph_range,fontsize=5)
plt.grid(axis="y")
plt.show()

