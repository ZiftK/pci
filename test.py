# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:46:43 2023

@author: ZiftK
"""

import numpy as np

import pandas as pd

from mathematics.pci import PCI, compare

def cuac(x):
    return x**2

#df = pd.read_csv("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv")

#pcys = PCI("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv",rounder = 50 ,offset =8)

#pcys = PCI("C:\\Users\\ZiftK\\Desktop\\TODO\\Python\\pci\\csvs\\sin.csv",rounder = 20 ,offset =15 )

# values to evaluate in functions
x_range = [x for x in np.arange(0.5,5.5,0.5)]

# real values
y_range = [np.sin(x) for x in x_range]

df = pd.DataFrame({"x":x_range,"y":y_range})

compare_range = [x for x in np.arange(0.5,5.1,0.1)]

offset_range = [off for off in range(10,21)]

error = compare(df, np.sin,compare_range,offset_range,30)

