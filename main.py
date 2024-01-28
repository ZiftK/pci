import pci as pci
import pandas as pd
from numpy import sin 

pd.set_option("display.max_rows",None)


# psys = pci.PCI("csvs/sin.csv",offset=5,rounder=50)

# psys.normalize(0.01)#TODO: Normalize bug

# print(psys.predict(-1.5,0.1))

df = pd.read_csv("csvs\sin.csv")

new_df = pci.uniform_data_range(df,sin,[x for x in range(1,10)],[y for y in range(1,20)])

new_df.to_csv("prueba-sin-1_75-0_5-sin_csv.csv",index = False)