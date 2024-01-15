import pci as pci

psys = pci.PCI("csvs/sin.csv")

psys.predict(3.2)