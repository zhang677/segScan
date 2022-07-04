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
f = open('../profile/'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'.csv','w')
headline = 'dataset,kernel'
for i in all_metric:
    headline += (','+i)
headline += '\n'
f.write(headline)
for input_matrix in input_matrices:
    print(input_matrix)
    prof_name = '../profile/'+input_matrix+'-prof.ncu-rep'
    my_context = ncu_report.load_report(prof_name)
    my_range = my_context.range_by_idx(0)
    my_action = my_range.action_by_idx(0)
    kernel_name = my_action.name()
    f.write(input_matrix+','+kernel_name)
    for i in all_metric:
        f.write(','+str(my_action.metric_by_name(i).as_double()))
    f.write('\n')

f.close()
    

