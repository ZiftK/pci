import pci as pci

psys = pci.PCI("csvs/sin.csv",offset=10)

print(psys.predict(0))