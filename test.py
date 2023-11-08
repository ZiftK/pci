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

df = pd.read_csv("C:\\Users\\ZiftK\\Desktop\\TODO\\Python\\pci\\csvs\\sin.csv")

pcys = PCI(df,rounder = 15,offset = 6)

val = 15

a = pcys.predict(val)
b = np.sin(val)
print(f"{abs((b-a)/b)*100}  % to a: {a} and b {b}")

print(pcys)