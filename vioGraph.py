import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('file.csv')
df2 = pd.read_csv('file2.csv')

plt.plot(np.arange(len(df['vio'])),list(df['vio']), color ="r", label = "Violations")
plt.plot(np.arange(len(df2['warn'])),list(df2['warn']), color = "y", label = "Warnings")
plt.xlabel("Frame")
plt.yticks(np.arange(max(df2['warn'])))
plt.ylabel("Count")
plt.legend()
plt.show()