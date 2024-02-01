import pci as pci
import pandas as pd
from numpy import sin 

pd.set_option("display.max_rows",None)


# psys = pci.PCI("csvs/sin.csv",offset=1,rounder=1)

# print(psys.predict( 50,0.1))
# print(psys.static_sp)

# df = pd.read_csv("csvs\sin.csv")

# new_df = pci.uniform_data_range(df,sin,[x for x in range(1,20)],[y for y in range(1,50)])

# new_df.to_csv("prueba-sin-1_75-0_5-sin_csv.csv",index = False)

# df = pd.read_csv("gdf1_geo.csv")