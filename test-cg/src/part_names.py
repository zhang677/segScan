"""
hardware = '2080'
cmd = 'cd ../profile/2080'
subprocess.call(cmd, shell=True)
cmd = 'ls > ../../src/part_names.txt'
subprocess.call(cmd, shell=True)
cmd = 'cd ../../src'
subprocess.call(cmd, shell=True)
"""
feature = '16'
feature_str = 3
f = open('part_names.txt','r')
lines = f.readlines()
outline = ''
for i in lines:
    name = i.split('.')[0]
    if name.split('-')[-1] == feature:
        outline += name[:-(5+feature_str)]+','
outline = outline[:-1]
f.close()
f = open('part_names.txt','w')
f.write(outline)
f.close()