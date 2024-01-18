# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""



import logging as lg
import mathematics.aop as aop 
import mathematics.dfop as dfop

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
        self.__dr = dfop.DataRange(data)

        # effective limits
        self.__le = None # effective lower limit
        self.__ue = None # effective upper limit

        # coeficients to save data range solution
        self.__coef = None
        # exponentes to save data range solution
        self.__exp = None


    #hd: Aux methods
    
    def is_inside_ef(self, point, column_name : str):
        '''
        If point is inside effective range return true, else return false
        '''
        try:#try check if inside effective range
            # is inside effecitve limits
            c1 = point >= self.dr.get_value(column_name, self.__le)
            c2 = point <= self.dr.get_value(column_name,self.__ue)
            return c1 and c2
        except TypeError:
            #if effective limits are null, set condition to false
            return False
        except KeyError:
            return False
        
    def extract_ef_df(self):
        '''
        Uses the effective lower and upper limits to extract 
        the data frame within the effective range.
        '''
        return self.dr.extract_df(self.__le,self.__ue)
    
    
        
    

    #hd: Train methods

    def train(self,point,offset,rounder):
        '''
        Train system to selected solve package around specified point
        '''

        # The train process take place in four stages:
        # calculating effective limits
        # calculating exponents, 
        # solving coefficients, 
        # and cleaning coefficients

        # Because the system can be trained in the dynamic or static range,
        # and each of these ranges contains its effective range,
        # SolvePackages are used to encapsulate all the variables
        # and functionalities of each super range (static and dynamic).
        # This way, we can independently control in each training phase in which
        # super range the system is being trained.

        self.__calc_ef_limits(point,offset) # calculating effective limits
        self.__calc_exp() # calculating exponents
        self.__solve(rounder) # solving coefficients
        self.__clear() # cleaning coefficients

    def __calc_ef_limits(self, point, offset):
        '''
        Calculate the effective limits for the SolvePackage using the specified point
        '''
        
        #The first step is to locate the pivot within the dataset,
        # which is simply the value within the dataset that is
        # closest to the point you want to approximate
        val = self.__dr.get_near_value("x",point)

        #Since the effective limits correspond to indices within
        # the dataset, the pivot must also be translated into an
        # index to locate it within the dataset
        val_indx = self.__dr.get_index("x",val)

        # The lower limit should not be less than zero, as it is
        # the minimum allowed value for an index. Therefore,
        # if the offset exceeds the range of values established
        # in the dataset, the effective range will be shortened,
        # and the effective lower index will be set to zero by default
        self.__le = max(0, val_indx - offset)

        # The same applies to the upper effective limit, as if the offset
        # from the pivot exceeds the range of values in the dataset,
        # the effective upper limit will be set to the maximum allowed
        self.__ue = min(val_indx + offset,self.__dr.rows_count()-1)

    def __calc_exp(self):
        '''
        Calculate the exponents used in the solution polynomial
        '''

        # The exponents are calculated using the length of the effective 
        # range as a parameter. A list of exponents is computed,
        # where each value represents the exponent of a term in the final 
        # polynomial (it represents the dimension of the monomial)

        #The super range contained in the assigned SolvePackage is used,
        # allowing us to obtain the exponents within the specifically
        # defined effective range for the point to be approximated.
        # If the point is within the static range, the static SolvePackage
        # will be used; otherwise, the dynamic one will be used

        self.__exp = [n for n in range(0,len(self.extract_ef_df()))]
    
    def __solve(self,rounder):
        '''
        Approximate the coefficients of the solution polynomial
        '''
        # To approximate the coefficients, it is necessary to solve a 
        # matrix (n, n), where 'n' is the number of data points used
        # for the approximation. The first step is to define this matrix
        m = list()

        # We subtract the effective data frame from the SolvePackage to
        # be used and evaluate at each of the 'x' column values in their 
        # respective rows.
        for x in self.extract_ef_df()["x"]:
            m.append(aop.valpow(  x, self.__exp))
        
        # ______ SOLVE ______
            
        # Define 'm' as a NumPy matrix object
        m = matrix(m)

        # Solve the matrix using the 'y' column of the effective data frame
        # as the expansion vector of the matrix
        self.__coef = linalg.solve(m, array(self.extract_ef_df()["y"]))

        # Round each polynomial coefficient using the rounder value
        self.__coef = round(self.__coef,rounder)

    def __clear(self):
        '''
        Delete Monomials with negligible coeficients from solution
        '''
        
        # In this list, the indices of each negligible coefficient
        # will be stored to delete them from the coefficients list
        del_index = list()
        
        # get index of negligible coeficients
        # iterate throught each round coeficient and get its index
        # for delete to polynomial
        for index, coef in enumerate(self.__coef):
            
            # add index with negligible coeficients
            if coef == 0:
                
                del_index.append(index)
        
        # This is done to generate polynomials as small as possible or to reduce noise
        self.__coef = delete(self.__coef,del_index)
        self.__exp = delete(self.__exp,del_index)


    def update_out_data(self,point, step = 0.5):
        '''
        Inserts a value outside the original data range, 
        offset by a value defined by 'step' towards the approximation point
        '''

        #TODO: document train loop
        while True:

            #pass the 'step' parameter through the 'abs' function to avoid 
            # using negative values (this would result in incorrect 
            # calculations as it would change the direction 
            # of the offset for extrapolation)
            step = abs(step)

            #Within this function, we only need to make one check.
            # To avoid defining too much code within conditionals
            # and with the intention of not repeating the same code too much,
            # we generate three variables that will be set within the 
            #conditionals to generalize the process using them.
            # Thus, the extrapolation will change direction with
            # the change of these variables

            # *last inside value
            # It refers to the nearest value to the desired extrapolation 
            # within the dynamic range 
            in_val = None 

            #* insert index 
            # It refers to the index where the new data will be inserted
            indx = None

            if point < self.__dr.get_value("x",0):

                # If the extrapolation point is to the left of the dynamic dataset,
                # we need to change the extrapolation direction. 
                # We achieve this by changing the sign of the 'step' to negative,
                # to decrease the extrapolation value
                step *= -1

                # In order to approximate a value outside the dynamic range, 
                # we need to know the last value within the range in the direction
                # of the extrapolation
                in_val = self.__dr.get_value("x",int(self.__le))
                
                # To insert the new extrapolated value within the dynamic range,
                # it is necessary to know on which side of the range it will be inserted.
                # We do this by establishing the insertion index. In this case,
                # the index is the effective lower limit, as the data is to the left
                # of the dynamic range
                indx = self.__le

            else:
                # It has already been verified that the point 
                # is not within the dynamic range, so if it is 
                # also not found on the left, the only possible 
                # option is that it is located on the right.

                # In order to approximate a value outside the dynamic range, 
                # we need to know the last value within the range in the direction
                # of the extrapolation
                in_val = self.__dr.get_value("x",self.__ue)

                # To insert the new extrapolated value within the dynamic range,
                # it is necessary to know on which side of the range it will be inserted.
                # We do this by establishing the insertion index. In this case,
                # the index is the effective upper limit, as the data is to the right
                # of the dynamic range
                indx = self.__ue
                

            # approximate the value outside the dynamic range using the dynamic SolvePackage
            out_val = self.apply_pol(in_val + step)

            # insert value in selected index
            self.__dr.insert(indx,in_val + step,"x")

            # set value in "y" column (aproximate value)
            self.__dr.set_value(indx,out_val,"y")

            # If the approximation point is within the
            # effective range, it means that we have
            # extrapolated the dataset enough to provide a result
            if self.is_inside_ef(point,"x"):
                return self.apply_pol(point)
    
    def apply_pol(self,point):
        '''
        Apply polinomial solution to specyfic point
        '''
        #* make prediction
        #pow point to each value in exponents array
        a = aop.valpow(float(point),self.__exp)
        # multiply each value in solve point exponents 
        # to each value in solve coefficients
        pdct = aop.amult(self.__coef,a)
        #return sum of array
        return sum(pdct)

    

    #hd: Properties

    @property
    def le(self):
        return self.__le
    
    @property
    def ue(self):
        return self.__ue

    @property
    def dr(self):
        return self.__dr

    @property
    def coef(self):
        return self.__coef

    @property
    def exp(self):
        return self.__exp    
    

    #hd: Object override

    def __str__(self):
        '''
        String object representation
        '''
        string = ""
        
        for index, coef in enumerate(self.__coef):
            string += f"{self.__coef[index]}*x^{self.__exp[index]}"
            string += "" if index == len(self.__coef)-1 else "+"
        return string.replace("e", "*10^")


class PCI:
    '''
    PCI (Polynomial with Coefficients Indeterminate) is a system 
    that approximates real data without apparent correlation through
    a Taylor series, a polynomial. This method requires the tabulation of data.

    Limitations
    -----------

    - Currently, the PCI system only works with single-variable correlation. 
    Its performance for multi-variable correlation is planned for the future.
    
    - The PCI system uses a large amount of memory to approximate values, 
    so setting a high offset and consequently a wide effective range can lead 
    to memory overflow.

    Properties
    ---------

    data -> The 'data' entered into the PCI system must be in tabular form. 
    Currently, a data frame or a path to extract it is supported. 
    If another type of data or an incorrect path is entered, an exception will be raised

    offset -> The 'offset' is the value used to control the amplitude of the effective range. 
    For practical and technical reasons, the entered value will be doubled to generate the final amplitude.
    **Important** This value will affect the accuracy of the final result.

    rounder -> The 'rounder' value controls how many decimals are considered 
    for calculations. Higher decimal precision incurs more computational cost. 
    Once the final polynomial is obtained, all coefficients rounded to zero 
    will be eliminated using 'rounder' as a parameter.
    **Important** This value will affect the accuracy of the final result.

    '''

    def __init__(self, data, **kwargs) -> None:

        # We initialize a SolvePackage for the data related 
        # to the dynamic range and another one for the data
        # related to the static range
        self.__ssp = SolvePackage(data)
        self.__dsp = SolvePackage(data)

        # set default offset
        self.__offset = kwargs.get("offset",5)
        # set default rounder
        self.__rounder = kwargs.get("rounder",15)
        
        # Testing variables
        self.__tst = None
        self.__tst2 = None
        


        
    
    
    def __train(self, point, solve_package : SolvePackage):
        '''
        
        '''
        solve_package.train(point,self.__offset,self.__rounder)


    
    def predict(self,point,ep_step = 0.5):
        '''
        Get aprox value from trained system
        '''
        
        point = round(point,5)
        
        # check if point is inside static effective data range or
        # dynamic effective data range

        # check if inside static effective data range
        if self.__ssp.is_inside_ef(point,"x"):
            return self.__ssp.apply_pol(point)

        # check if inside dynamic effective data range
        elif self.__dsp.is_inside_ef(point,"x"):
            return self.__dsp.apply_pol(point)
        
        # it is inside static limits?
        elif self.__ssp.dr.is_inside(point,"x"):

            #train system inside static limits
            self.__train(point, self.__ssp)

            # apply polinomial solution to static solve package
            return self.__ssp.apply_pol(point)

        # It has been previously verified that the point to approximate 
        # is outside the dynamic effective range and the 
        # static effective range. It has also been confirmed to be 
        # outside the static range, so the only available options are 
        # that it is within the dynamic range or it is outside all ranges.

        # check if point is inside dynamic range
        in_dynamic = self.__dsp.dr.is_inside(point,"x")
        
        # train system in dynamic solve package
        self.__train(point, self.__dsp)

            
        if in_dynamic: #* if point is in dynamic range
            
            # apply polinomial solution to dynamic solve package
            return self.__apply_pol(point,self.__dsp)
        
        # In case the point is outside the dynamic range, 
        # the dynamic range should be updated by providing 
        # feedback until the desired value is reached; 
        # this is done by the update_dynamic function
        return self.__dsp.update_out_data(point,ep_step)


    def normalize(self,step = 0.1):
        '''
        Flat dynamic data frame using normal as difference value
        '''

        
        cur_val = round( self.__dsp.dr.get_value("x",0) + step,self.__rounder)
        indx = 1

        while True:

            if self.__dsp.dr.is_inside(cur_val,"x"):

                pdct_val = self.predict(cur_val)

                self.__dsp.dr.insert(indx,cur_val,"x")

                self.__dsp.dr.set_value("y",indx,pdct_val)

            if not self.__dsp.dr.is_inside(cur_val + step,"x"):
                break

            cur_val += step
            indx += 1

        print(self.__dsp.dr.extract_df(0,self.__dsp.dr.rows_count()))

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
    '''
    Calculate relative error from real value and his aproximate value
    '''
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