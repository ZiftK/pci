import pci as pci
import pandas as pd

pd.set_option("display.max_rows",None)


psys = pci.PCI("csvs/sin.csv",offset=10,rounder=50)

psys.normalize()

print(psys.predict(-1.5,0.1))