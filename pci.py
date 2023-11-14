# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""


import dfop
import logging as lg
import aop 

from numpy import array, arange, matrix, linalg, round, delete, dot, diag




class PCI:
    
    '''
    TODO
    -----

    Rangos
    ------

    Es necesario considerar el uso de variables para almacenar 
    el data frame del rango efectivo y del rango dinámico.
    Bajo nuevas consideraciones, se decidío trabajar únicamente sobre
    el rango estático, definiendo el rango efectivo a través de sus índices.
    Esto es posible gracias a que el rango efectivo solo es utilizado
    para entrenar el modelo y la comprobación del punto se lleva a cabo mediante
    los índices del rango efectivo. Por lo tanto, no es necesario almacenarlo, ya
    que una vez entrenado el sistema bajo ese rango, no desea volver a entrenarse
    en el mismo, a menos que se hagan predicciones fuera del rango y después dentro
    de él.\n
    En cuanto al rango dinámico, se desea eliminar para trabajar sobre el rango
    estático, expandiendolo de ser necesario.

    Módulo de manipulación de data frames
    --------------------------------

    Es necesario crear un módulo que manipule los data frames con las operaciones
    requeridas por el módulo PCI, como la expanción del rango estático tanto en
    runtime como en un archivo para su permanencia.

    Normalización de datos
    ----------------------
    Se necesita normalizar los datos para aumentar la precisión de los mismos.
    Esto pretende conseguirse realizando interpolaciones que completen los datos

    '''
    def __init__(self, df_path : str,**kwargs):
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
        self.__df = dfop.read_csv(df_path) # original data frame

        #* Calc mean diff data
        self.__mean_diff = dfop.mean_diff(self.__df,"x")
        
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

        #* Init coefficients
        self.__coefficients = None

        #* Save exp aray
        # These exponents are calculated considering a vector space of dimension 'n',_
        # where 'n' is the number of input data. This vector space corresponds to a polynomial of_
        #  degree (n-1). The exponents are stored in a numpy array because all operations will be_
        # performed using arrays
        self.__exp = None

        self.__tst = None
        self.__tst2 = None
        
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
        edf = self.__calc_effective_df()

        #solve ecuation system using effective data frame data (solve coefficients)
        self.__solve(edf)

        #clear useless coefficients
        self.__clear()

    def __calc_exp(self, degree):
        
        #inflog: calculating exponents
        self.__log.info("Calculating exponents...")

        #calculate exponents
        self.__exp = [n for n in range(0,degree)]

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

        self.__li = 0   #lower limit
        self.__ls = len(self.__df)-1#upper limit
        pass

    def __calc_effective_limits(self,point):
        '''
        Calculate effective limits on data
        '''
        
        # check if point is inside static range
        if self.__li < point and point < self.__ls:

            #get nearest value to point (pivot)
            cp = dfop.near_val(self.__df,"x",point)
            
            #inflog: calculating effective limits
            self.__log.info(f"Calculating effective limits using -- {cp} -- as central point")
            
            #get cp index
            cp_index = dfop.get_index(self.__df,"x",cp)
            

            self.__ci = max(cp_index-self.__offset,self.__li) #lower effective limit
            self.__cs = min(cp_index+self.__offset,self.__ls) #upper effectiev limit

            #deblog: log ci and cs
            self.__log.debug(f"Set ci as {self.__ci} and cs as {self.__cs}")

            #inflog: ci and cs setted
            self.__log.info("Ci and Cs setted"+"\n"*2)

        else:

            pass



    def __calc_effective_df(self):
        '''
        Set effective range as data frame 
        '''
        
        self.__log.info("Calculating effective data frame...")

        #deblog: print ci index
        self.__log.debug(f"Ci index set as {self.__ci}")  

        #deblog: print cs index
        self.__log.debug(f"Cs index set as {self.__cs}")  

        #inflog: setting effective data frame
        self.__log.info("Setting effective data frame")

        # get effective range as data frame
        edf = dfop.segment(self.__df.sort_values(by="x"),self.__ci,self.__cs)

        print("------------",len(edf),"--",self.__ci,"--",self.__cs)
        # calc exponents
        self.__calc_exp(len(edf))

        #inflog: effective data frame setted
        self.__log.info("Effective data frame setted\n\n")

        #return effective data frame
        return edf

    def __solve(self, edf : dfop.DataFrame):
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
        for x in edf["x"]:
            m.append(aop.valpow(  x,self.__exp))
        
        #deblog: show pows
        self.__log.debug(f"\nPows: {m}\n")

        #inflog: solving
        self.__log.info("Solving")
        
        # ______ SOLVE ______
        m = matrix(m)
        

        #deblog: print solve matrix
        self.__log.debug(f"Matrix : {m.shape[0]},{m.shape[1]} \n Extention: {len(edf['y'])}")
        
        # save coefficients
        self.__coefficients = linalg.solve(m, array(edf["y"]))
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

    def normalize(self, normal = None):

        # if normal is empty, set to mean data diff
        if not normal:
            normal = self.__mean_diff

        #set initial value as first value of "x" column
        initial_val = self.__df["x"][0]

        #set final value as last value of "x" column
        final_val = self.__df["x"][len(self.__df)-1]

        #
        for val in arange(initial_val,final_val + self.__mean_diff, self.__mean_diff):
            
            if not ( self.__df["x"].isin([val]).any()):

                predict_val = self.predict(val)

                predict_index = dfop.get_index(self.__df,"x",val-self.__mean_diff) + 1

                self.__df = dfop.insert(self.__df,"x",predict_index,val)

                self.__df.loc[predict_index,"y"] = predict_val

                print("SIIIIIIIIIII  ")
                    


        
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

        a = aop.valpow(float(point),self.__exp)

        pdct = aop.amult(self.__coefficients,a)
        
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
        self.__rounder = max(1,value)


if __name__ == "__main__":
    
    

    pass