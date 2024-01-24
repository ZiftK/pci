import pci as pci
import pandas as pd

pd.set_option("display.max_rows",None)


psys = pci.PCI("csvs/sin.csv",offset=5,rounder=50)

psys.normalize(0.01)#TODO: Normalize bug

print(psys.predict(-1.5,0.1))