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
datasets = []
ratios = []
min_datasets = {}
for i in range(actions):
    min_datasets[i] = []
for i in range(dataset_num):
    times = []
    for l in range(actions):
        if (l!=0) and (l!=1) and (l!=2) : # Depends on where cusparse results are recorded
            times.append(df.loc[i*actions+l].Time)
    min_action = times.index(min(times))+3
    min_actions.append(min_action)
    min_datasets[min_action].append(df.loc[i*actions].dataset)
    ratios.append(times[0]/min(times))
hist = []
for i in range(actions):
    hist.append(min_actions.count(i))
print(hist)
print(np.mean(ratios))
print(np.std(ratios))
print(gmean(ratios))
