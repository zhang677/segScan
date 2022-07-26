#!/bin/bash

MATRICES_DIR=/home/nfs_data/datasets/sparse_mat
#MATRICES_DIR=/home/nfs_data/zhanggh/segScan/test-cg/data
MODE=1
RESULTS_DIR=/home/nfs_data/zhanggh/segScan/test-cg/profile
export LD_LIBRARY_PATH="/home/eva_share/opt/cuda-11.6/lib64:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 ./test.py $MATRICES_DIR $MODE $RESULTS_DIR $1 $2