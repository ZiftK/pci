from pci import PCI

psys = PCI("csvs/sin.csv")

a = psys.predict(0)

print(a)