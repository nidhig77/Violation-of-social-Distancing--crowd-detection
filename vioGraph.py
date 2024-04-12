import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('file.csv')

print(list(df['vio']))

plt.plot(np.arange(len(df['vio'])),list(df['vio']))
plt.xlabel("Frame")
plt.yticks(np.arange(max(df['vio'])))
plt.ylabel("Violations")
plt.show()