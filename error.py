
from numpy import array, arange,mean
from pci import PCI
from pandas import DataFrame

def calc_error(ideal_values : array, real_values : array,mult = 100) -> array:
    
    return array(abs((ideal_values-real_values)/ideal_values)*mult)

def pci_vs_local_func_error(pci : PCI, local_func, values_range : arange):
    
    ideal_values = array([local_func(x) for x in values_range])
    real_values = array([pci.predict(x) for x in values_range])
    
    return calc_error(ideal_values, real_values)

def error_to_plot(pcys : PCI, local_func, values_range : arange):
    
    error = pci_vs_local_func_error(pcys, local_func, values_range)
    
 
    print(max(error))
    print(mean(error))
    edf = DataFrame(error)
    
    return edf.set_index(values_range)