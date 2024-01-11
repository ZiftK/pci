# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""



import logging as lg
import aop 
import dfop

from numpy import array, arange, matrix, linalg, round, delete, dot, diag


class SolvePackage:
    '''
    The Resolution Package is a class designed to encapsulate a range of 
    data along with its corresponding limits and values used in the PCI approximation. 
    Essentially, it is a class that enables PCI to locate the values of its dynamic or static data.

    Properties
    ---------

    df -> It refers to the feeding DataRange. From this DataRange, 
    the data for the approximation will be taken.

    le -> Effective Lower Limit [PCI Method].

    ue -> Effective Upper Limit [PCI Method].

    coef -> Coefficients obtained within the interval defined 
    by the effective lower limit and the effective upper limit [PCI Method].

    exp -> Exponents obtained within the interval defined 
    by the effective lower limit and the effective upper limit [PCI Method].

    '''

    def __init__(self, data : dfop.DataFrame):

        # data range object
        self.df = dfop.DataRange(data)

        # effective limits
        self.le = None # effective lower limit
        self.ue = None # effective upper limit

        # coeficients to save data range solution
        self.coef = None
        # exponentes to save data range solution
        self.exp = None

    def is_inside_ef(self, point, column_name : str):
        '''
        If point is inside effective range return true, else return false
        '''
        try:#try check if inside effective range
            # is inside effecitve limits
            return point >= self.df.get_value(column_name, self.le) and point <= self.df.get_value(column_name,self.ue)
        except TypeError:
            #if effective limits are null, set condition to false
            return False
        except KeyError:
            return False
        

class PCI:


    def __init__(self, data, **kwargs) -> None:
        '''
        PCI System was designed to predict values using data
        '''

        # initialize solve packages
        self.__ssp = SolvePackage(data)
        self.__dsp = SolvePackage(data)

        # set default offset
        self.__offset = kwargs.get("offset",5)
        # set default rounder
        self.__rounder = kwargs.get("rounder",15)
        
        # Testing variables
        self.__tst = None
        self.__tst2 = None
        
    def __train(self, solve_package: SolvePackage):
        '''
        Train system to specyfic effective data frame
        '''
        self.__calc_exp(solve_package)
        self.__solve(solve_package)
        self.__clear(solve_package)

    def __calc_exp(self,solve_package : SolvePackage):
        '''
        Calculate exponentes to solve data
        '''
        # calculate exponents to solve data
        solve_package.exp = [n for n in range(0,len(solve_package.df))]
    
    def __solve(self, solve_package : SolvePackage):
        '''
        Train system to interpolate data (get polynomial)

        Returns
        -------
        None.

        '''
        # matrix to resolve
        m = list()

        # evaluate each x value into each matrix function line
        for x in solve_package.dfq["x"]:
            m.append(aop.valpow(  x, solve_package.exp))
        
        # ______ SOLVE ______
        m = matrix(m)

        # save coefficients
        solve_package.coef = linalg.solve(m, array(solve_package.df["y"]))
        solve_package.coef = round(solve_package.coef,self.__rounder)

    def __clear(self, solve_package : SolvePackage):
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
        for index, coef in enumerate(solve_package.coef):
            
            # add index with despicable coeficients
            if coef == 0:
                
                del_index.append(index)
        
        # This is done to generate polynomials as small as possible or to reduce noise
        solve_package.coef = delete(solve_package.coef,del_index)
        solve_package.exp = delete(solve_package.exp,del_index)


        
    def __apply_pol(self,point, solve_package : SolvePackage):
        '''
        Apply polinomial solution to specyfic point
        '''
        #* make prediction
        #pow point to each value in exponents array
        a = aop.valpow(float(point),solve_package.exp)
        # multiply each value in solve point exponents 
        # to each value in solve coefficients
        pdct = aop.amult(solve_package.coef,a)
        #return sum of array
        return sum(pdct)
    
    def __update_dynamic(self):
        
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
    
    def predict(self,point):
        '''
        Get aprox value from trained system
        '''
        
        point = round(point,5)
        
        # check if point is inside static effective data range or
        # dynamic effective data range

        # check if inside static effective data range
        if self.__ssp.is_inside_ef(point,"x"):
            return self.__apply_pol(point,self.__ssp)

        # check if inside dynamic effective data range
        elif self.__dsp.is_inside_ef(point,"x"):
            return self.__apply_pol(point,self.__dsp)
        
        # is inside static limits
        elif self.__ssp.df.is_inside(point,"x"):

            #train system inside static limits
            self.__train(self.__ssp)

            # apply polinomial solution to static solve package
            return self.__apply_pol(point,self.__ssp)

        while True: #* train loop


            in_dynamic = self.__dsp.df.is_inside(point,"x")
                
            if in_dynamic: #* if point is in dynamic range
                
                self.__train(self.__dsp)
                return self.__apply_pol(point,self.__dsp)
                break #break while
    
    
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