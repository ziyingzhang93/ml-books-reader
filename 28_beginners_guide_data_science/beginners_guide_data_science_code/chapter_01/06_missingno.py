import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
msno.matrix(Ames, sparkline=False, fontsize=20)
plt.show()
