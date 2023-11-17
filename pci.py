# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""


import dfop
import logging as lg
import aop 

from numpy import array, arange, matrix, linalg, round, delete, dot, diag


class PCI:

    def __init__(self, data_path) -> None:
        

        # Get data frames
        self.__df = dfop.read_csv(data_path)
        self.__ddf = self.__df.copy()

        



if __name__ == "__main__":
    
    

    pass