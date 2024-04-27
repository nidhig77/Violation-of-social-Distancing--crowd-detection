import pandas as pd
import numpy as np
ours = pd.read_csv("results.csv")
true_res = pd.read_csv("result_org.csv")

count = 0

ours = ours.values.tolist()
ours = np.array(ours)
ours.astype(str)
while "nan, 0.0, 1.0" in ours:
    ours.remove("nan, 0.0, 1.0")
true_res = true_res.values.tolist()
true_res = np.array(true_res)
true_res.astype(str)
while "nan, 0.0, 1.0" in true_res:
    ours.remove("nan, 0.0, 1.0")
print(ours)
for i in range(len(true_res)):
    if ours[i].all()==true_res[i].all():
        count += 1

print("accuracy="+str(float(count)/float(len(ours))))