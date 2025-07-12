import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../boost_converter_dataset.csv')

# 电子器件 MOSFET
cap = df['P_mosfet']
print(cap.describe())
cap.plot.hist(bins=30, alpha=0.7, grid=True)
plt.title('P_mosfet Distribution')
plt.xlabel('P_mosfet')
plt.show()


# 电子器件 Capacitor
cap = df['P_capacitor']
print(cap.describe())
cap.plot.hist(bins=30, alpha=0.7, grid=True)
plt.title('P_capacitor Distribution')
plt.xlabel('P_capacitor')
plt.show()


# 电子器件 Diode
cap = df['P_diode']
print(cap.describe())
cap.plot.hist(bins=30, alpha=0.7, grid=True)
plt.title('P_diode Distribution')
plt.xlabel('P_diode')
plt.show()


# 电子器件 Inductor
cap = df['P_inductor']
print(cap.describe())
cap.plot.hist(bins=30, alpha=0.7, grid=True)
plt.title('P_inductor Distribution')
plt.xlabel('P_inductor')
plt.show()
