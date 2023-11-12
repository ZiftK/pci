# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""


import pandas as pd
import logging as lg

from numpy import array, arange, matrix, linalg, round, delete, dot, diag




class PCI:
    
    def __init__(self, df : pd.DataFrame,**kwargs):
        '''
        
        '''
        #sp: -------- set log --------
        
        self.__log = lg.getLogger(__name__)

        if len(self.__log.handlers) == 0:
            # console handler
            hnd = lg.StreamHandler()
            # set format
            hnd.setFormatter(lg.Formatter('%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'))
            # set log handler
            self.__log.addHandler(hnd)
            # set log level
            self.__log.setLevel(lg.DEBUG)
            # log setted
            self.__log.info("Log setted!!!"+"\n"*2)

        #inflog: initialize
        self.__log.info("Initialize..."+"\n"*2)

        #sp: -------- input values --------

        #inflog: setting optional values
        self.__log.info("Setting optional values..."+"\n"*2)
        # This parameter will be used to set the cardinality of the effective range
        self.__offset = kwargs.get("offset",10)
        # This parameter is used to determine how many decimal places to round the coefficient value
        self.__rounder = kwargs.get("rounder",10)

        #sp: -------- Assigned values --------
        
        self.__log.info("Setting initial values..."+"\n"*2)
        #* Save data frame
        self.__df = df # original data frame
        
        #* Static range limits
        self.__li = None # represents the smallest value within static range
        self.__ls = None # represents de highest value within static range
        self.__calc_static_limits() # set static range limits

        #* Effective range limits
        self.__ci = None # represents the samllest value within effective range 
        self.__cs = None # represents the highest value within effective range

        #* Dynamic range limits
        self.__di = None # represents the smallest value within dynamic range
        self.__ds = None # represents the highest value within dynamic range

        #* Init dynamic data frame
        self.__ddf = None # represents dinamyc range as data frame

        #* Init effective data frame
        self.__edf = None # represents effective range as data frame

        #* Init coefficients
        self.__coefficients = None

        #* Save exp aray
        # These exponents are calculated considering a vector space of dimension 'n',_
        # where 'n' is the number of input data. This vector space corresponds to a polynomial of_
        #  degree (n-1). The exponents are stored in a numpy array because all operations will be_
        # performed using arrays
        self.__exp = None
        
        pass
    

    #hd: Close methods

    def __train(self, pivot):
        '''
            Train system to make predicts
        '''

        #inflog: training
        self.__log.info("Training..."+"\n"*2)  #log
        
        #calculate effective limits
        self.__calc_effective_limits(pivot)

        #calculate effective data frame
        self.__calc_effective_df()

        #solve ecuation system using effective data frame data (solve coefficients)
        self.__solve()

        #clear useless coefficients
        self.__clear()

    def __calc_exp(self, degree):
        
        #inflog: calculating exponents
        self.__log.info("Calculating exponents...")

        #calculate exponents
        self.__exp = array([n for n in range(0,degree)])

        #deblog: calculated as
        self.__log.info(f"Exponents setted as {self.__exp}")

        #inflog: exponents setted
        self.__log.info("Exponents setted"+"\n"*2)
        pass

    def __calc_static_limits(self):
        '''
        Calculate static limits on data
        '''
        #TODO: Falta considerar el desorden de un data frame para asignar los límites del rango estático
        self.__log.info("Calculating static limits...\n\n")

        self.__li = self.__df["x"][0]   #lower limit
        self.__ls = self.__df["x"][len(self.__df)-1] #upper limit
        pass

    def __calc_effective_limits(self,point):
        '''
        Calculate effective limits on data
        '''

        #inflog: calculating effective limits
        self.__log.info(f"Calculating effective limits using -- {point} -- as central point")

        self.__ci = self.__df_near_val(point - self.__offset/2) #lower limit
        self.__cs = self.__df_near_val(point + self.__offset/2+0.1) #upper limit

        #deblog: log ci and cs
        self.__log.debug(f"Set ci as {self.__ci} and cs as {self.__cs}")

        #inflog: ci and cs setted
        self.__log.info("Ci and Cs setted"+"\n"*2)

    def __calc_dynamic_limits(self):
        '''
            Useless
        '''

        #TODO: add dynamic limits
        pass

    def __df_near_val(self,x):
        '''
        Get the near value to x in to internal data frame
        
        Parameters
        ----------
        x -> value to search
        
        Returns
        --------
        Near value to x param
        '''
        return self.__df['x'].iloc[(self.__df['x'] - x).abs().idxmin()]

    def __calc_effective_df(self):
        '''
        Set effective range as data frame 
        '''
        
        self.__log.info("Calculating effective data frame...")

        #* Get ci index

        #inflog: get ci index
        self.__log.info("Getting Ci index")

        # get ci index list
        ci_indx = self.__df.index[self.__df['x'] == self.__ci]
        # get first ci index
        ci_indx = ci_indx[0]

        #deblog: print ci index
        self.__log.debug(f"Ci index set as {ci_indx}")  

        #* Get cl index

        #inflog: get cs index
        self.__log.info("Gettin Cs index") 

        #get cl index list
        cs_indx = self.__df.index[self.__df["x"] == self.__cs]+1
        # get first cl index
        cs_indx = cs_indx[0]

        #deblog: print cs index
        self.__log.debug(f"Cs index set as {cs_indx}")  

        #inflog: setting effective data frame
        self.__log.info("Setting effective data frame")

        # get effective range as data frame
        edf = self.__df.sort_values(by="x")[ci_indx:cs_indx]

        # if data frame is empty (effective range outside static range)
        if edf.empty:
            #TODO: add extrapolation options

            #inflog: pass to extrapolation
            self.__log.info("Pass to extrapolation")
            # extrapolation
            raise Exception(
                "Extrapolation exception. Value out of range",
                f"Ci_index: {ci_indx} - Cl_index: {cs_indx}"
                )
        
        else:

            # set effective range
            self.__edf = edf

            #inflog: effective data frame setted
            self.__log.info("Effective data frame setted\n\n")

        # del index
        del cs_indx, ci_indx

        # calc exponents
        self.__calc_exp(len(self.__edf))

    def __solve(self):
        '''
        Train system to interpolate data (get polynomial)

        Returns
        -------
        None.

        '''
        
        #inflog: solving coefficients
        self.__log.info("Solving coefficients...")
        # matrix to resolve
        m = list()

        #inflog: adjusting lines
        self.__log.info("Adjusting lines")
        # evaluate each x value into each matrix function line
        for x in self.__edf["x"]:
            m.append(x**self.__exp)
        
        #inflog: solving
        self.__log.info("Solving")

        # ______ SOLVE ______
        m = matrix(m)

        #deblog: print solve matrix
        self.__log.debug(f"Matrix : {m.shape[0]},{m.shape[1]} \n Extention: {len(self.__edf['y'])}")
        
        # save coefficients
        self.__coefficients = linalg.solve(m, array(self.__edf["y"]))
        self.__coefficients = round(self.__coefficients,self.__rounder)

        #deblog: coefficients
        self.__log.debug(f"Coefficients set as {self.__coefficients}")

        #inflog: coefficients solved
        self.__log.info("Coefficients solved"+"\n"*2)

        
        
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
 
    
    #hd: User methods

    def reset(self,df : pd.DataFrame):
        '''
        Reset PCI with new data frame
        '''
        self = PCI(df)
        
        
    def predict(self,point):
        '''
        Get predict value to specific parameters

        Returns
        --------
        Predict value
        '''

        #inflog: predicting
        self.__log.info("Predicting...\n\n")


        #if value is not in effective range
        if (not ( self.__ci and self.__cs)) or (not (point >= self.__ci and point <= self.__cs)):
            
            #deblog: retrainning
            self.__log.debug("Training again...\n\n")
            self.__train(point)
            pass

        # apply polinomial function [a_1, a_2, a_3, ... , a_n]*[value**(n-1),value**(n-2), ... , value**1, value**0]
        # and sum all monomial
        pdct = self.__coefficients*(point**self.__exp)
        
        return sum(pdct)
    
    def __str__(self):
        '''
        String object representation
        '''
        string = ""
        
        for index, coef in enumerate(self.__coefficients):
            string += f"{self.__coefficients[index]}*x^{self.__exp[index]}"
            string += "" if index == len(self.__coefficients)-1 else "+"
        return string.replace("e", "*10^")
 
    #hd: Getters

    #* Range limits
    @property
    def effective_limits(self):
        return self.__ci,self.__cs

    @property
    def static_limits(self):
        return self.__li,self.__ls

    @property
    def dynamic_limits(self):
        return self.__di,self.__ds
    
    #* Optional values


    @property
    def offset(self):
        '''
        Effecitve range cardinality
        '''
        return self.__offset
    
    @offset.setter
    def offset(self,value):
        '''
        Set offset to value if is grerater than 1, else set to 1
        '''
        self.__offset = max(value,1)

    @property
    def rounder(self):
        '''
        Decimal round count
        '''
        return self.__rounder
    
    @rounder.setter
    def rounder(self,value):
        '''
        Set rounder to value if is greater than 0, else set to 0
        '''
        self.__rounder = max(0,value)


if __name__ == "__main__":
    
    

    pass