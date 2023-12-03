# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""



import logging as lg
import aop 
import dfop

from numpy import array, arange, matrix, linalg, round, delete, dot, diag


class PCI:

    def __init__(self, data, **kwargs) -> None:
        

        if type(data) == str:
            # *Get data frames
            self.__df = dfop.read_csv(data) # static data frame
            

        elif type(data) == dfop.DataFrame:
            self.__df = data

        else:
            raise Exception("Invalid data type. Must be string path or pandas data frame.")

        self.__ddf = self.__df.copy() # dynamic data frame

        #* Static limits
        self.__li = 0 # lowwer limit
        self.__ls = len(self.__df)-1 # upper limit

        #* Dynamic limits
        self.__di = self.__li # upper limit
        self.__ds = self.__ls # lowwer limit

        self.__ci = None
        self.__cs = None
        
        #* Dynamic effective limits
        self.__dci = None # Lowwer limit
        self.__dcs = None # upper limit
        
        #* Static effective limits
        self.__sci = None # loweer limit
        self.__scs = None # upper limit

        #* Dynamic calc values
        self.__dcoefficients = None # dynamic coefficients
        self.__dexp = None  # dynamic exponents

        #* Static calc values
        self.__scoefficients = None
        self.__sexp = None

        self.__coefficients = None
        self.__exp = None

        self.__offset = kwargs.get("offset",5)
        self.__rounder = kwargs.get("rounder",15)

        self.__mean_diff = dfop.mean_diff(self.__df,"x")
        
        self.__tst = None
        self.__tst2 = None
        
    def __train(self, edf: dfop.DataFrame):
        '''
        Train system to specyfic effective data frame
        '''
        self.__calc_exp(len(edf))
        self.__solve(edf)
        self.__clear()
        pass

    def __calc_exp(self, degree):
        '''
        Calculate exponents with specyfic degree
        '''
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
        pivot_index = dfop.get_index(df,"x",pivot)

        # *set effective limits
        # set effective limits between static range
        self.__ci = max(ref_i,pivot_index - self.__offset)
        self.__cs = min(ref_s,pivot_index + self.__offset)

        #* get effective data frame
        edf = dfop.segment(df,self.__ci,self.__cs)
        
        self.__tst = edf

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
        self.__tst2 = pdct
        #return sum of array
        return sum(pdct)
    
    def predict(self,point):
        '''
        Get aprox value from trained system
        '''
        
        point = round(point,5)
        print(point)
        # is inside static limits
        in_static = point <= self.__df["x"][self.__ls] and point >= self.__df["x"][self.__li]

        print(f"{self.__df['x'][self.__ls]} -- {self.__df['x'][self.__li]}")
        while True: #* train loop
            # while will be breake it if
            # predict point is inside any range,
            # else, the system iterate throught dynamic
            # range to update it to point in it

            # is inside dynamic limits 
            in_dynamic = point < self.__ddf["x"][self.__ds] and point > self.__ddf["x"][self.__di]

            try:#try check if inside effective range
                # is inside effecitve limits
                in_effective = point < self.__df["x"][self.__cs] and point > self.__df["x"][self.__ci]
            except TypeError:
                #if effective limits are null, set condition to false
                in_effective = False
            except KeyError:
                in_effective = False
                
            
            if in_effective: #* if point is inside effective range
                
                break #break while

            elif in_static: #* if point is inside n static range
 
                self.__train_in_limits(self.__df,self.__li,self.__ls,point)
                break #break while
            

            else: #* else
                # if point is outside effective and static ranges,
                # should be inside or outside dynamic range, wathever
                # system train it in dynamiic range

                # train in dinamic range
                self.__train_in_limits(self.__ddf,self.__di,self.__ds,point)
                
                if in_dynamic: #* if point is in dynamic range
                    
                    break #break while
                
                # init predict point
                predict_point = None
                # aux values
                limit = None
                step = None
                df_0 = None
                
                #* predict one step outside dynamic range
                # if point is left to dynamic range
                if point < self.__df["x"][self.__di]:
                    # set limit as lower dynamic limit
                    limit = self.__di
                    # set step as negative mean diff
                    step = -1*(0.1) #!Valor experimental
                    pass
                
                # if point is right to dynamic range
                else:
                    # set limit as upper dynamic limit
                    limit = self.__ds
                    # set step as mean diff
                    step = 0.1 #!Valor experimental
                    pass
                
                # apply point prediction with outside step
                predict_point = self.__ddf["x"][limit] + step
                # get extrapolation value
                extrapol_val = self.__apply_pol(predict_point)
                # create new data frame
                df_0 = dfop.DataFrame({"x":[predict_point],"y":[extrapol_val]})
                # concat order
                concat = [df_0,self.__ddf] if point < self.__df["x"][self.__di] else [self.__ddf,df_0]
                # concat data frames
                self.__ddf = dfop.concat(concat, axis=0, ignore_index=True)
                
                pass
        
        
        return self.__apply_pol(point)
    
    
    def normalize(self,normal = None):
        '''
        Flat dynamic data frame using normal as difference value
        '''
        if normal == None:# If optional normal is null
            normal = self.__mean_diff # set normal to data frame mean diff
        
        #set initial value to iterate
        initial = self.__df["x"][self.__li]
        #set final value to iterate
        final = self.__df["x"][self.__ls] + normal
        
        #index to insert new predict value
        insert_index = 0
        
        #init new data frame
        df = self.__df.copy()
        
        #iterate throgth init range
        for x in arange(initial,final,normal):
            
            if not x in self.__df["x"]:# if predict value is not in static data frame
                
                y = self.predict(x)
                dfop.insert(df,"x",insert_index,x)
                self.__ddf["y"][insert_index] = y
                
            insert_index += 1
        
        #set static data frame to new data
        self.__df = df
    
    def __str__(self):
        '''
        String object representation
        '''
        string = ""
        
        for index, coef in enumerate(self.__coefficients):
            string += f"{self.__coefficients[index]}*x^{self.__exp[index]}"
            string += "" if index == len(self.__coefficients)-1 else "+"
        return string.replace("e", "*10^")

def pcit_ov(data, offset_range, rounder, values_range)-> dict:
    '''
    Iterate through the offset range
    and set the PCI system for each offset in the range. 
    Then, approximate each value in the values range using each 
    one of the offsets set
    '''

    # aprox_values is a dictionary that stores numpy arrays. 
    # This is because for each offset in the offset range, a 
    # list of approximate values will be calculated.
    # Each key of the ‘aprox_values’ dictionary represents 
    # the offset that will be calculated, and its content 
    # represents the output PCI predicted values from the 
    # ‘values_range’ as inputs.

    aprox_values : dict = dict() # aproximate values dictionary
    current_values : list = list() # ccurrent PCI predicted values for each step

    # iterate throught offset range.
    # For each offset in 'offset_range'
    # a list of PCI predicted values will
    # be calculate.
    for current_off in offset_range: 

        # debug message
        print(f"\n\nPCI offset variable** current offset: {current_off}")

        # init pci system with offset as current offset
        pcys = PCI(data, rounder = rounder, offset = current_off)
        
        # iterate throught values range.
        # For each value in the ‘values_range’, 
        # a PCI predicted value will be calculated.
        for value in values_range: 

            print(f"\t current value: {value}")

            # aproximate values
            current_values.append(pcys.predict(value))

        # vectorize and save values
        aprox_values[current_off] = array(current_values)

        # clear current values
        current_values.clear()

        del pcys # delete pci system

    return aprox_values # return aproximate values

def pcit_rv(data, offset, rounder_range, values_range)-> dict:
    '''
    Iterate through the rounder range and set the PCI
    system for each rounder in the range. Then, aproximate
    each value in the values range using each one of the
    rounder set
    '''

    aprox_values : dict = dict() # aproximate values dictionary
    current_values : list = list() 

    for current_round in rounder_range: # iterate throught offset range

        # debug message
        print(f"\n\nPCI rounder variable** current round: {current_round}")

        # init pci system
        pcys = PCI(data,rounder = current_round, offset = offset)

        for value in values_range: # iterate throught values range

            print(f"\t current value: {value}")

            # aproximate values
            current_values.append(pcys.predict(value))

        # vectorize values and save it
        aprox_values[current_round] = array(current_values)

        # clear current values
        current_values.clear()

        del pcys # delete pci system

    return aprox_values # return aproximate values

def relative_error(real_val,aprox_val):
    return 100*abs((real_val-aprox_val)/real_val)

def compare(data, real_function,values_range, offset, rounder):

    # sp: ------------- Vars -------------

    # variables to set pci evaluate function
    offset_iter, rounder_iter = True, True
    # pci evaluate function
    pci_func = None
    # real function values
    real_values = list()
    # aproximate pci values
    aprox_values = list()
    # relative error
    error = 0
    # errors dictionary
    error_dict = dict()

    # sp: ------------- Check iterables -------------

    try:# check if offset is iterable
        iter(offset)
    except TypeError:# save it
        offset_iter = False

    try:# chacj if rounder is iterable
        iter(rounder) 
    except TypeError:# save it
        rounder_iter = False

    # Check whether offset or rounder is iterable because the 
    # function only supports one of the two being iterable

    # sp: ------------- Def pci function -------------

    # if offset is iterable and rounder is not
    if offset_iter and not rounder_iter:

        # set pci evaluate function as pci offset variable
        pci_func = pcit_ov

    # if rounder is iterable and offset is not
    elif rounder_iter and not offset_iter:

        # set pci evaluate function as pci rounder variable
        pci_func = pcit_rv
    else:
        raise Exception("Offset and rounder cant vary at the same time")
    
    # sp: ------------- Calculate values -------------

    # calculate aprox values
    aprox_values = pci_func(data, offset,rounder,values_range)

    #* Calculate real values
    for value in values_range:
        real_values.append(real_function(value))

    #sp: ------------- Calculate error -------------

    # vectorice lists
    real_values =  array(real_values)

    for key in aprox_values.keys():
        error = relative_error(real_values,aprox_values[key])
        error_dict[key] = error



    return error_dict



if __name__ == "__main__":
    
    try:
        iter(2)
        print("iterable")
    except TypeError:
        print("no iterable")
    
    pass