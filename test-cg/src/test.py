#!/usr/bin/python3

import os
import sys
import subprocess

def execute_cmd(cmd):
    cmd = ' '.join(cmd)
    print(cmd)
    subprocess.call(cmd, shell=True)

matrices_dir = sys.argv[1]
mode = sys.argv[2]
results_dir = sys.argv[3]
prof_dir = '/home/nfs_data/zhanggh/segScan/test-cg/profile'

input_matrices = []
if matrices_dir == '/home/nfs_data/datasets/sparse_mat':
    f = open('names.txt','r')
    input_matrices = f.readline().split(',')
    input_matrices.sort()
    f.close()
elif matrices_dir == '/home/nfs_data/zhanggh/segScan/test-cg/data':
    f = open('part_names.txt','r')
    input_matrices = f.readline().split(',')
    input_matrices.sort()
    f.close()

print(input_matrices)
feature_size = '128'
if mode == '0':
    print('==Check Mode==')
    for input_matrix in input_matrices:
        input_matrix_dir = os.path.join(matrices_dir, input_matrix, input_matrix+'.mtx')
        cmd = ['./test',input_matrix_dir, feature_size, mode]
        print(' '.join(cmd))
        subprocess.run(cmd)
elif mode == '1':
    print('==Test Mode==')
    ncu = 'sudo LD_LIBRARY_PATH="/home/eva_share/opt/cuda-11.6/lib64:$LD_LIBRARY_PATH" /home/eva_share/opt/cuda-11.6/bin/ncu'
    f = open('prof_names.txt', 'r')
    profs = f.readline()
    for input_matrix in input_matrices:
        input_matrix_dir = os.path.join(matrices_dir, input_matrix, input_matrix+'.mtx')
        cmd = [ncu, '-o', '../profile/'+input_matrix+'-prof','-f','--metrics',profs,'--target-processes', 'all','./test',input_matrix_dir, feature_size, mode]
        execute_cmd(cmd)
        cmd = ['sudo','chmod','777','../profile/'+input_matrix+'-prof.ncu-rep']
        execute_cmd(cmd)
        #subprocess.run(cmd)
else:
    raise NotImplementedError
# Check

