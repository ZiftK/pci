
# import numpy array
from numpy import array
from numpy import arange

import time

def exec_time(func):
    def wrapper(*args, **kwargs):
        intl = time.time()
        rtn = func(*args, **kwargs)
        fnl = time.time()
        tt = fnl - intl
        print(f"The function {func.__name__} was executed in {tt:.6f} seconds.")
        return rtn
    return wrapper

