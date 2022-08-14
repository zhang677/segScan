import pandas as pd
import numpy as np
from scipy.stats import gmean
import sys
fileName = sys.argv[1]
actions = int(sys.argv[2])
k = int(sys.argv[3])
#fileName = '../tables/3090/0708-003206-3090.csv'
df = pd.read_csv(fileName)
lines = df.shape[0]
#actions = 60
dataset_num = (int)(lines/actions)
datasets = []
ratios = []
for i in range(dataset_num):
    times = []
    cuspt = 0
    for l in range(actions):
        if (l<3):
            cuspt += df.loc[i*actions+l].Time
        else : # Depends on where cusparse results are recorded
            times.append(df.loc[i*actions+l].Time)
    ratios.append(times[0]/times[k])
    datasets.append(df.loc[i*actions].dataset)

print(max(ratios))
print(datasets[ratios.index(max(ratios))])
print(np.mean(ratios))
print(np.std(ratios))
print(gmean(ratios))
print(len([i for i  in ratios if i >1]))