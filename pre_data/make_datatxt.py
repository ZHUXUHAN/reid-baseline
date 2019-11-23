
_pids = []
with open('/home/zxh/reid-baseline/data/samplers/number_less4.txt', 'r') as f:
    _pids_list = eval(f.readline().strip())
for i in range(len(_pids_list)):
    _pids.append(int(_pids_list[i]))
for i in range(4768, 6116):
    _pids.append(i)

with open("/home/zxh/reid-baseline/data/samplers/number_newtrain_less4.txt", 'w') as f:
    f.write(str(_pids))
