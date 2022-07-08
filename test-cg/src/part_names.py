"""
hardware = '2080'
cmd = 'cd ../profile/2080'
subprocess.call(cmd, shell=True)
cmd = 'ls > ../../src/part_names.txt'
subprocess.call(cmd, shell=True)
cmd = 'cd ../../src'
subprocess.call(cmd, shell=True)
"""
f = open('part_names.txt','r')
lines = f.readlines()
outline = ''
for i in lines:
    outline += (i.split('.')[0])[:-5]+','
outline = outline[:-1]
f.close()
f = open('part_names.txt','w')
f.write(outline)
f.close()