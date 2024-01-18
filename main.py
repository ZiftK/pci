import pci as pci

psys = pci.PCI("csvs/sin.csv",offset=10,rounder=50)

psys.normalize()

print(psys.predict(-1.5,0.1))

import pandas as pd

# DataFrame de ejemplo
data = {'Columna1': [1, 2, 3],
        'Columna2': ['a', 'b', 'c']}
df = pd.DataFrame(data)

# Nueva fila que deseas agregar
nueva_fila = {'Columna1': 4, 'Columna2': 'd'}

# Usando loc para agregar la nueva fila
df.loc[len(df)] = nueva_fila

print(df)