
from pandas import read_csv, concat
from pandas import DataFrame, Series

from numpy import arange


def min_value(df : DataFrame, column_name : str):
    '''
    Return min valuue from column in data frame
    '''

    return df[column_name].min()

def max_value(df: DataFrame, column_name : str):
    '''
    Return max value from column in data frame
    '''

    return df[column_name].max()

def get_val_from_index(df : DataFrame, column_name : str ,index : int):
    '''
    Return index row value from passed column
    '''

    return df[column_name].iloc[index]

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
        
    
    def extract_df(self, initial_index : int, final_index : int):
        '''
        Return a sub data frame from original data frame, from initial to final index
        '''
        return segment(self.__df,initial_index,final_index)
    
    def min_val_from(self, column_name : str):
        '''
        Return min value from specifyc column
        '''

        return min_value(self.__df,column_name)

    def max_val_from(self, column_name : str):
        '''
        Return max value from specfyc column
        '''
        
        return max_value(self.__df,column_name)
    
    def get_value(self, column_name : str, index : int):
        '''
        Return index value from column name
        '''

        return get_val_from_index(self.__df,column_name,index)
    
    def get_near_value(self, column_name : str, val):
        '''
        Search nearest value to *val* in column and return it
        '''
            
        return near_val(self.__df, column_name, val)
    
    def sort_by(self, column_name : str):
        '''
        Sort original data frame using values from passed column
        '''

        self.__df = self.__df.sort_values(by=column_name)

    @DeprecationWarning
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
                insert(df,"x",insert_index,x)
                self.__ddf["y"][insert_index] = y
                
            insert_index += 1
        
        #set static data frame to new data
        self.__df = df