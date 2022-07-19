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
means = []
stds = []
gmeans = []
for k in range(actions-3):
    ratios = []
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
        ratios.append(times[k]/min(times))
    means.append(np.mean(ratios))
    stds.append(np.std(ratios))
    gmeans.append(gmean(ratios))

epsilon = 0.01
min_mean = min(gmeans)
print([(i,means[i],stds[i],gmeans[i]) for (i,d) in enumerate(gmeans) if abs(min_mean-d) < epsilon])