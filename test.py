# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:46:43 2023

@author: ZiftK
"""

import numpy as np

import pandas as pd

from pci import PCI

import error as er

def cuac(x):
    return x**2

#df = pd.read_csv("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv")

#pcys = PCI("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv",rounder = 300 ,offset =5 )

pcys = PCI("C:\\Users\\ZiftK\\Desktop\\TODO\\Python\\pci\\csvs\\sin.csv",rounder = 300 ,offset =5 )

val = 3
a = pcys.predict(val)
b = np.sin(val)

print(f"\n\n{abs((b-a)/b)*100}  % to a: {a} and b {b}")

print("\n\n")
print(pcys)


pcys.normalize(0.1)
