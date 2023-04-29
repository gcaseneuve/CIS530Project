import pandas as pd
import numpy as np
import math
df = pd.read_csv("cluster1.csv")
df['Average Magnitude'].fillna(0, inplace= True)

eq_cnt = []
eq_avg = []
year = []
for i in range(1480, 2010, 10):
    filterIdx = np.where((df['Year'] >  i-10) & (df['Year'] <= i ))
    if filterIdx[0].size == 0:
        eq_avg.append(0)
        eq_cnt.append(0)
        year.append(i) 
        continue
    eq_avg.append(df.loc[filterIdx]['Average Magnitude'].mean())
    eq_cnt.append(df.loc[filterIdx]['Count'].sum())
    year.append(i)
dataset = {'Year': year, 'Avg Mag': eq_avg}
df = pd.DataFrame(dataset)
df.to_csv('reg_data.csv', index= False)
