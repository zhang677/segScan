import pandas as pd
import numpy as np
from scipy.stats import gmean
import sys
fileName = sys.argv[1]
actions = int(sys.argv[2])
#fileName = '../tables/3090/0708-003206-3090.csv'
df = pd.read_csv(fileName)
lines = df.shape[0]
#actions = 60
dataset_num = (int)(lines/actions)
min_actions = []
ratios = []
ratios_cu = []
min_datasets = {}
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
    #min_datasets[min_action].append(df.loc[i*actions].dataset)
    ratios.append(times[0]/min(times))
    ratios_cu.append(cuspt/min(times))
hist = []
for i in range(actions):
    hist.append(min_actions.count(i))
print(hist)
print(np.mean(ratios))
print(np.std(ratios))
print(gmean(ratios))
print(np.mean(ratios_cu))
print(np.std(ratios_cu))
print(gmean(ratios_cu))