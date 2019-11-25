import numpy as np
import json
dist_npy = './result/submit/dist_65.npy' #生成的dist矩阵，距离越小越相似
dist = np.load(dist_npy,allow_pickle=True)

rank = np.argsort(dist)
rank0 = rank[:,0]
print(len(rank0))
print(len(set(rank0)))

query_name_npy = './result/submit/query_name.npy'
gallery_name_npy = './result/submit/gallery_name.npy'

query_name = np.load(query_name_npy,allow_pickle=True) # query 图片名字的list（与计算dist时 保持一致）
gallery_name = np.load(gallery_name_npy,allow_pickle=True) # gallery 图片名字list（与计算dist时 保持一致）

dist_cp = dist.copy()
dist_cp.sort(1)
dist_r1 = dist_cp[:,0]
rank1 = np.argsort(dist_r1)
dist_r1.sort()
print(dist_r1)
print(dist[rank1[1]][rank0[rank1[1]]])

flags = np.zeros(len(gallery_name))
result = {}
target_json = './11241.json' #提交结果文件

thr = dist_r1[int(len(rank1)*0.85)]
for i in range(len(dist)):
    if i%50 == 0:
        print(i)
        print(sum(flags))
    query_index = rank1[i]
    gallery_list = np.argsort(dist)[query_index]
    dist_i = dist[query_index]
    
    result[query_name[query_index]]=[]
    
    num = 0
    first=True
    for g in gallery_list:
        if flags[g] == 1:
            first=False
            continue
        if first:
#             if i < int(len(query_name)*0.85):
            flags[g] = 1
            first = False
        if dist_i[g] < thr:
            flags[g] = 1
        result[query_name[query_index]].append(gallery_name[g])
        num += 1
        if num == 200:
            break
with open(target_json,"w") as f:
    json.dump(result,f)
