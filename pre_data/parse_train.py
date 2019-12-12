with open("/home/zxh/train_list.txt")  as f:
    lines = f.readlines()

id_dict = {}
for line in lines:
    line = line.strip().split(' ')
    imgname = line[0].split('/')[1]
    imgid = line[1]
    if int(imgid) not in id_dict:
        id_dict[int(imgid)] = [imgname]
    else:
        id_dict[int(imgid)].append(imgname)

id_long_dict = {}
for k, v in id_dict.items():
    id_long = len(v)
    if id_long not in id_long_dict:
        id_long_dict[id_long] = [k]
    else:
        id_long_dict[id_long].append(k)

id_less_4 = []
for k, v in id_long_dict.items():
    if len(v)<4:
        for vv in v:
            id_less_4.append(vv)
with open("less_4.txt", 'w') as f:
    f.write(str(id_less_4))
