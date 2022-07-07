import pandas as pd
fileName = '../tables/7_6_1_36.csv'
df = pd.read_csv(fileName)
lines = df.shape[0]
actions = 32
dataset_num = (int)(lines/actions)
min_actions = []
datasets = []
min_datasets = {}
for i in range(actions):
    min_datasets[i] = []
for i in range(dataset_num):
    times = []
    for l in range(actions):
        if (l!=1) and (l!=2) and (l!=3) : # Depends on where cusparse results are recorded
            times.append(df.loc[i*actions+l].Time)
    min_action = times.index(min(times))+3
    min_actions.append(min_action)
    min_datasets[min_action].append(df.loc[i*actions].dataset)
hist = []
for i in range(actions):
    hist.append(min_actions.count(i))
print(hist)