#Jan 20243

import pandas as pd
import numpy as np

df = pd.read_csv('./sim3.csv', header=None, index_col=False).to_numpy()
#print(df.shape)
x = np.array(df)
num = 1

for i in range(0,8000,200):
  #print(i)
  data = x[i:(i+200),:]
  print(data.shape)
  np.savetxt(f'sim3_{num}.txt', data, delimiter=',')
  
  num += 1