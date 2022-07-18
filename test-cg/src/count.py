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
min_actions = []
ratios = []
ratios_cu = []
min_datasets = {}
datasets = []
for i in range(actions):
    min_datasets[i] = []

for i in range(dataset_num):
    times = []
    cuspt = 0
    for l in range(actions):
        if (l<3):
            cuspt += df.loc[i*actions+l].Time
        else : # Depends on where cusparse results are recorded
            times.append(df.loc[i*actions+l].Time)
    min_action = times.index(min(times))+3
    min_actions.append(min_action)
    datasets.append(df.loc[i*actions].dataset)
    #min_datasets[min_action].append(df.loc[i*actions].dataset)
    ratios.append(times[k]/min(times))
    ratios_cu.append(cuspt/min(times))
hist = []
for i in range(actions):
    hist.append(min_actions.count(i))
print(hist)
print(k)
print(max(ratios))
print(datasets[ratios.index(max(ratios))])
print(np.mean(ratios))
print(np.std(ratios))
print(gmean(ratios))
print(len([i for i  in ratios_cu if i >1]))
print([datasets[i] for (i,d) in enumerate(ratios_cu) if d > 1])
print([i for i in ratios_cu if i > 1])
print(max(ratios_cu))
print(datasets[ratios_cu.index(max(ratios_cu))])
print(np.mean(ratios_cu))
print(np.std(ratios_cu))
print(gmean(ratios_cu))