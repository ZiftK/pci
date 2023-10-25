# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""


import pandas as pd

from numpy import array, arange, matrix, linalg

class PCI:
    
    def __init__(self, df : pd.DataFrame):
        
        #save data frame
        self.__df = df
        #save length
        self.__n = len(df)
        #save exp aray
        self.__exp = array([(self.__n-k-1) for k in range(self.__n)])
        
        self.__coefficients = None
        
        self.__solve()
        
        pass
    
    def __solve(self):
        
        m = list()
        for x in self.__df["x"]:
            m.append(x**self.__exp)
        
        m = matrix(m)
        
        solutions = list(self.__df["y"])
        
        self.__coefficients = array(linalg.solve(m,solutions))
        
    def reset(self,df : pd.DataFrame):
        
        self = PCI(df)
        
        
    def predict(self,value):
        
        pdct = self.__coefficients*(value**self.__exp)
        
        return sum(pdct)
    
 
    

if __name__ == "__main__":
    
    

    #get data frame from csv
    df : pd.DataFrame = pd.read_csv("C:\\Users\\alvar\\OneDrive\\Desktop\\Stg\\Projects\\PCI\\csvs\\sin.csv")

    #get data frame length
    n = len(df)

    #init m as list, this will be a matrix
    m = list()

    #init exp array
    exp = [(n-k-1) for k in range(n)]
    exp = array(exp)

    #for each data in x pow to each exponent in exp
    for x in df["x"]:
        m.append(x**exp)

    #switch to matrix
    m = np.matrix(m)

    #f(x) list
    res = list(df["y"])

    #solve matrix
    final_res = np.linalg.solve(m, res)

    #aprox value
    predict_value = 4

    #predict variable
    predict = final_res*(p**exp)
    predict = sum(final_rese)