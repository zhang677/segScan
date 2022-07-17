# segScan
Cooperative group on segScan
# Warning
Change the directory of datasets, names to your own path
# Run
Go to `test-cg/src`.
To do the checking
```
make clean TARGET=check
make TARGET=check
./run.sh
```
To do the testing
```
make clean TARGET=test
make TARGET=test
./run.sh
```
# Observation
1. Cooperative group doesn't necessarily improve the performance (7_6_1_36.csv, 7_7_3_34.csv)

| Dataset     | Macro | Cooperative Group |
| :---:       |    :----:   |    :---: |
| 192bit      | 25.9552 ms  | 28.8416 ms  |
| 12month1    | 2588.6400 ms| 2386.9184 ms|

2. Optimal area instead of point (7_7_3_40.csv, 7_7_5_50.csv)
Execution time difference is with in 0.5%

| Optimal block_factor | Execution Time |
| ----------- | ----------- |
| 1/16 | 2332.5920 ms |
| 1 | 2345.6064 ms |

3. The optimal area depends on datasets (7_7_1_12.csv, 7_7_3_40.csv)

| Dataset     | group_factor | tail_factor | thread_per_block | block_factor|
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 192bit      |     2       |       2     | 256 | (1/2,2) |
| 12month1    |     4       |       5     | 128 | (1/16,1)|

4. Results of acceleration brought by tuning

ebpr

| Hardware | mean | std | geomean | N | 
| :---:    | :----:| :---:| :---:| :---:| 
| 3090 | 1.506 | 0.315 | 1.477 | 128 |
| 3090 | 1.476 | 0.301 | 1.449 | 64 |
| 2080 | 1.245 | 0.187 | 1.232 | 128 |
| 2080 | 1.245 | 0.193 | 1.232 | 64 |
| V100 | 1.538 | 0.449 | 1.481 | 128 |
| V100 | 1.524 | 0.439 | 1.469 | 64 |

rbpr

| Hardware | mean | std | geomean | max | N | 
| :---:    | :----:| :---:| :---:| :---:|  :---:| 
| 3090 | 2.448 | 0.886 | 2.277 | 4.384 | 128 |
| 2080 | 1.853 | 0.533 | 1.777 | 3.679 | 128 |
| V100 | 1.663 | 0.452 | 1.602 | 2.903 | 128 |

5. Best result vs. cuSPARSE

ebpr

| Hardware | mean | std | geomean | N | 
| :---:    | :----:| :---:| :---:| :---:| 
| 3090 | 0.371 | 0.151 | 0.342 | 128 |
| 3090 | 0.396 | 0.161 | 0.371 | 64 |
| 2080 | 0.292 | 0.130 | 0.263 | 128 |
| 2080 | 0.319 | 0.111 | 0.303 | 64 |
| V100 | 0.314 | 0.122 | 0.295 | 128 |
| V100 | 0.420 | 0.182 | 0.397 | 64 |

rbpr

| Hardware | mean | std | geomean | max | faster| N | 
| :---:    | :----:| :---:| :---:| :---:| :---:| :---:| 
| 3090 | 0.525 | 0.155 | 0.499 | 2.256 | cop20k_A | 128 |
| 2080 | 0.336 | 0.083 | 0.327 | 1.130 | cop20k_A | 128 |
| V100 | 0.391 | 0.133 | 0.374 | 2.682 | cop20k_A | 128 |

6. Dynamic vs. static

ebpr

| Hardware | mean | std | geomean | N | Best static |
| :---:    | :----:| :---:| :---:| :---:| :---:| 
| 3090 | 1.042 | 0.048 | 1.041 | 128 | (5,256,5,1,32) |
| 3090 | 1.040 | 0.055 | 1.039 | 64 | (5,256,5,1,16) |

rbpr

| Hardware | mean | std | geomean | N | Best static |
| :---:    | :----:| :---:| :---:| :---:| :---:| 
| 3090 | 1.155 | 0.0356 | 1.124 | 128 | (3,256,3,1,2) |
| 2080 | 1.118 | 0.288  | 1.095 | 128 | (2,256,3,1,2) |
| V100 | 1.182 | 0.459  | 1.137 | 128 | (3,256,3,1,2) |


