import ncu_report
import datetime
matrix_names = 'part_names.txt'
f = open(matrix_names,'r')
input_matrices = f.readline().split(',')
input_matrices.sort()
f.close()
f = open('prof_names.txt','r')
all_metric = f.readline().split(',')
f.close()
now = datetime.datetime.now()
wrongf = open('wrong_names.txt','w')
f = open('../tables/'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'.csv','w')
launch_metric = ['launch__block_dim_x','launch__block_dim_y','launch__grid_dim_x','launch__grid_dim_y','launch__block_size', 'launch__grid_size']
nick_names = ['L1_Red','L1_Global','L1_Hit','L2_Load','L2_Red','L2_Hit','DRAM_Read','DRAM_Write','Waves_per_SM','ActiveWarps_per_SM','Time','SOL_Mem','SOL_SM','BlockDimX','BlockDimY','GridDimX','GridDimY','BlockSize','GridSize']
headline = ''
all_metric.extend(launch_metric)
for i in all_metric:
    headline += (nick_names[all_metric.index(i)]+',')
headline += 'dataset,kernel,action'
headline += '\n'
f.write(headline)
Wrong = False
for input_matrix in input_matrices:
    print(input_matrix)
    prof_name = '../profile/'+input_matrix+'-prof.ncu-rep'
    my_context = ncu_report.load_report(prof_name)
    my_range = my_context.range_by_idx(0)
    for j in range(my_range.num_actions()):
        my_action = my_range.action_by_idx(j)
        kernel_name = my_action.name()
        for i in all_metric:
            if my_action.metric_by_name(i) is None:
                Wrong = True
                if i == 0:
                    wrongf.write(input_matrix+',')
            else:
                f.write(str(my_action.metric_by_name(i).as_double())+',')
        if not Wrong:
            f.write(input_matrix+','+kernel_name+','+str(j))
            f.write('\n')
        Wrong = False

f.close()
wrongf.close()

