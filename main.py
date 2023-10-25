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

df : DataFrame = read_csv("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv")

pcys = PCI(df)

val_range = arange(9,12,0.5)

edf = err.error_to_plot(pcys, sin,val_range)

pcys.right_feecback(2,0.5)

plt.plot(edf)
plt.xticks(val_range,fontsize=7)
plt.grid()
plt.show()