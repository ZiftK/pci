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

    def __calc_exp(self, degree):
        
        #inflog: calculating exponents
        self.__log.info("Calculating exponents...")

        #calculate exponents
        self.__exp = [n for n in range(0,degree)]

        #deblog: calculated as
        self.__log.info(f"Exponents setted as {self.__exp}")

        #inflog: exponents setted
        self.__log.info("Exponents setted"+"\n"*2)
        pass
    
    def __solve(self, edf : dfop.DataFrame):
        '''
        Train system to interpolate data (get polynomial)

        Returns
        -------
        None.

        '''
        

        # matrix to resolve
        m = list()

        # evaluate each x value into each matrix function line
        for x in edf["x"]:
            m.append(aop.valpow(  x,self.__exp))
        
        # ______ SOLVE ______
        m = matrix(m)
        
        
        # save coefficients
        self.__coefficients = linalg.solve(m, array(edf["y"]))
        self.__coefficients = round(self.__coefficients,self.__rounder)

    def __clear(self):
        '''
        Delete Monomials with despicable coeficients

        Returns
        -------
        None.

        '''
        
        #inflog: clear coefficients
        
        self.__log.info("Cleaning coefficients...")
        # Index list to delete
        del_index = list()
        
        # get index of despicable coeficients
        # iterate throught each round coeficient and get his index
        # for delete to polynomial
        for index, coef in enumerate(self.__coefficients):
            
            # add index with despicable coeficients
            if coef == 0:
                
                #deblog: deleted - index
                self.__log.debug(f"Coefficient - {index} - deleted")
                del_index.append(index)
        
        # This is done to generate polynomials as small as possible or to reduce noise
        self.__coefficients = delete(self.__coefficients,del_index)
        self.__exp = delete(self.__exp,del_index)

        #inflog: coefficients cleaned
        self.__log.info("Coefficients cleaned")



if __name__ == "__main__":
    
    

    pass