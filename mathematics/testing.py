
# import numpy array
from numpy import array
from numpy import arange

def calc_error(real_value,aprox_value):
    '''
    Calculate relative error from real and aprox value

    Summary
    --------
    100*abs(real - aprox)/real
    '''
    return abs(real_value-aprox_value)*100/real_value

def get_func_values(real_func,aprox_func,range:arange):
    '''
    Compare real_func return value and aprox func return value, and calculate
    his relative error

    Params
    --------
    real_func -> real funciton\n
    aproxi_func -> aproximate function\n
    range -> range of values to perform test\n


    Returns
    --------
    Lists of evaluate values to range param\n
    real_val , aprox_val : numpy arrays
    '''

    real_val : list = list() # real values
    aprox_val : list = list() # aprox values

    #iterate trought each value in range
    for value in range:
        
        real_val.append(real_func(value)) # calc real value
        aprox_val.append(aprox_func(value)) # calc aprox value
    
    # convert list values to numpy arrays
    real_val = array(real_val)
    aprox_val = array(aprox_val)

    # return pack
    return  real_val, aprox_val
    
