
from pandas import read_csv, concat
from pandas import DataFrame, Series



def segment(df : DataFrame,initial_index : int, final_index : int) -> DataFrame:
    '''
    Segment data frame from *initial_index* to *final_index* and return it
    '''
    return df[initial_index:final_index]

def near_val(df : DataFrame, column_name : str, val):
    '''
    Search nearest value to *val* in column
    '''

    return df['x'].iloc[(df[column_name] - val).abs().idxmin()]

def get_index(df : DataFrame, column_name : str, val):
    '''
    Return index of value *val* in column *column_name*
    '''

    return df.index[df[column_name] == val][0]

def mean_diff(df : DataFrame, column_name : str)-> float:
    '''
    Returns mean diff of data in column *column_name*
    '''

    # Calcula la diferencia entre cada fila y la fila anterior
    diff = df[column_name].diff()

    # Calcula el promedio de las diferencias
    return diff.value_counts().idxmax()

def insert(df: DataFrame,column_name : str, index : int, value : float):
    '''
    
    '''

    # Inserta el valor 6 en la fila 2 de la columna y desplaza los valores hacia abajo
    df.loc[index:, column_name] = df.loc[index:, column_name].shift(1)
    df.loc[index, column_name] = value



class DataRange:

    def __init__(self,df: DataFrame) -> None:
        
        self.__df = None # data frame

        if type(df) == DataFrame: #try get data frame
            self.__df = df
        elif type(df) == str:
            self.__df = read_csv(df)
        else:# if param type are not data frame or string raise exception
            raise TypeError("A data frame or string path are expected")
        

        self.__ci = None #upper effective limit
        self.__cs = None #lower effective limit
        self.__edf = None # effective data frame
        self.__exp = None # resolve exponents
    

    def __calc_exp(self):
        '''
        Calculate exponents array
        '''
        self.__exp = [x for x in range(0,len(self.__edf))]

    @property
    def ci(self):
        '''
        Return ci limit
        '''
        return self.__ci
    
    @property
    def cs(self):
        '''
        Return cs limit
        '''
        return self.__cs
    
    @property
    def cis(self):
        '''
        Return ci and cs limit as tuple
        '''
        return self.__ci, self.__cs
    
    @property
    def exp(self):
        '''
        Return exponents list
        '''
        return self.__exp