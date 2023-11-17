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

    def __init__(self, data_path, **kwargs) -> None:
        

        # *Get data frames
        self.__df = dfop.read_csv(data_path)
        self.__ddf = self.__df.copy()

        self.__li = 0
        self.__ls = len(self.__df)-1

        self.__di = self.__li
        self.__ds = self.__ls

        self.__ci = None
        self.__cs = None

        self.__coefficients = None

        self.__offset = kwargs.get("offset",5)
        self.__rounder = kwargs.get("rounder",15)

    def __train(self, edf: dfop.DataFrame):

        self.__calc_exp(len(edf))
        self.__solve(edf)
        self.__clear()
        pass

    def __calc_exp(self, degree):
        
        #calculate exponents
        self.__exp = [n for n in range(0,degree)]
    
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

    
    def __train_in_limits(self, df : dfop.DataFrame, ref_i : int, ref_s : int, point):
        '''
        Set ci and cs limits using data frame, ref_i and ref_s index as reference
        around the point
        '''
        # get nearest value to *point* in column x from static data frame
        pivot = dfop.near_val(df,"x",point)

        # *set effective limits
        # set effective limits between static range
        self.__ci = max(ref_i,pivot - self.__offset)
        self.__cs = min(ref_s,pivot + self.__offset)

        #* get effective data frame
        edf = dfop.segment(df,self.__ci,self.__cs)

        #* train in effective data frame
        self.__train(edf)
        
    def __apply_pol(self,point):
        '''
        Apply polinomial solution to specyfic point
        '''
        #* make prediction
        #pow point to each value in exponents array
        a = aop.valpow(float(point),self.__exp)
        # multiply each value in solve point exponents 
        # to each value in solve coefficients
        pdct = aop.amult(self.__coefficients,a)
        #return sum of array
        return sum(pdct)
    
    def predict(self,point):

        # is inside static limits
        in_static = point < self.__ls and point > self.__li

        # is inside dynamic limits 
        in_dynamic = point < self.__ds and point > self.__di

        try:#try check if inside effective range
            # is inside effecitve limits
            in_effective = point < self.__cs and point > self.__ci
        except AttributeError:
            #if effective limits are null, set condition to false
            in_effective = False

        # initi effective data frame to None
        edf = None
        rtn = None
        
        while True:
            # while will be breake it if
            # predict point is inside any range,
            # else, the system iterate throught dynamic
            # range to update it to point in it
        
            if in_effective:
                
                break #break while

            elif in_static:
                
                self.__train_in_limits(self.__df,self.__li,self.__ls,point)
                break #break while
            

            else:
                # if point is outside effective and static ranges,
                # should be inside or outside dynamic range, wathever
                # system train it in dynamiic range
                
                # train in dinamic range
                self.__train_in_limits(self.__ddf,self.__di,self.__ds,point)
                
                if in_dynamic:
                    
                    #if point is inside dynamic range
                    break #break while
                
                
                
                pass
        
        
        return self.__apply_pol(point)
        
if __name__ == "__main__":
    

    pass