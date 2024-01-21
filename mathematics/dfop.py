
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

    
    df.loc[index:, column_name] = df.loc[index:, column_name].shift(1)
    df.loc[index, column_name] = value

def soft_insert(df: DataFrame, row : dict, index : int)-> DataFrame:
    '''
    Adds a row with the specified index, shifting the replacement 
    row and all others to the right.

    The specified value will be assigned to the specified column; 
    all other columns will have a default value.

    Returns
    --------
    DataFrame with new row inserted
    '''

    # To insert a new row into the data frame in a way that
    # its length can be manipulated, it is necessary to split
    # the data frame into two parts: the segment called 'up'
    # and the segment called 'down'

    # The 'up' segment comprises from the beginning of
    # the DataFrame to the insertion position
    up = segment(df,0,index)

    # The 'up' segment comprises from the insertion
    # position to the end of the DataFrame
    down = segment(df,index, len(df))

    # Since the 'ignore_index' option must be set to
    # 'True' to insert a row from a dictionary, 
    # we need to manually increment the indices
    # of the 'down' segment by one
    down.index += 1

    # We create a new series from the dictionary
    # passed as an argument to insert it as the
    # new row. We set its 'name' to the value 
    # of the insertion index
    new_row = Series(row)

    # We add the new row to the 'up' segment
    new_df = up._append(new_row,ignore_index = True)
    # Join all segments
    new_df = new_df._append(down)

    # Return join data frame
    return new_df

def set_value(df : DataFrame, column_name : str, index : int, value : float):
    '''
        Set value of specified column and index into value passed
    '''
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
    
    @DeprecationWarning
    def insert(self, index, value, column_name):
        '''
        
        '''

        insert(self.__df,column_name,index,value)

    def soft_insert(self, row : dict, index : int):
        '''
        Adds a row with the specified index, shifting the replacement 
        row and all others to the right.

        The specified value will be assigned to the specified column; 
        all other columns will have a default value.
        '''
        self.__df = soft_insert(self.__df,row,index)

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
    
    def is_inside(self, value, column_name : str):
        '''
        Check if value is inside data range of specifyc column
        '''
        return value >= self.min_val_from(column_name) and value <= self.max_val_from(column_name)
    
    def is_in_column(self,value, column_name : str):
        '''
        Check if passed value is in passed column
        '''
        return value in self.__df[column_name].values
    
    def get_mean_diff(self, column_name : str):
        '''
        Return the mean diff of data in data frame from specyfic column
        '''

        return mean_diff(self.__df, column_name)
    
    def get_value(self, column_name : str, index : int):
        '''
        Return index value from column name
        '''

        return get_val_from_index(self.__df,column_name,index)
    
    def set_value(self,column_name :str, index, value):
        '''
        Set value of specified column and index into value passed
        '''

        set_value(self.__df,column_name,index, value)

    
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

    def rows_count(self):
        '''
        Return rows count
        '''
        return len(self.__df)
    
    def get_index(self,column_name : str, val):
        '''
        Return first match index of value passed
        '''
        return get_index(self.__df,column_name,val)
    
    