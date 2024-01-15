import pci as pci

psys = pci.PCI("csvs/sin.csv")

print(psys.predict(3.2))