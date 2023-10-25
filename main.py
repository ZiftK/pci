# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:26:47 2023

@author: alvar
"""

from pci import PCI
from pandas import read_csv
from pandas import DataFrame, Series

from numpy import arange, array, sin

import matplotlib.pyplot as plt

df : DataFrame = read_csv("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv")

pcys = PCI(df)

pdct = [pcys.predict(val) for val in arange(1.5,12.5)]

pdct = array(pdct)

real_data = [sin(val) for val in arange(1.5,12.5)]

real_data = array(real_data)

error = abs((real_data-pdct)/real_data)*100

edf = DataFrame(error)

edf = edf.set_index(arange(1.5,12.5))

plt.plot(edf)
plt.xticks(arange(1.5,12.6))
plt.yticks(arange(0,0.045,0.005))
plt.grid()
plt.show()