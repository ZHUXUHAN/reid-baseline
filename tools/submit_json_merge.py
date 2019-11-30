# encoding:utf-8
import numpy as np
import json
import json, os
import numpy as np
def write_json2(dist,query_name,gallery_name):
    num_query,num_gallery = dist.shape
    print(dist.shape)
    dist_line = dist.reshape(-1)
    print(dist_line.shape)
    dist_sorted_indices = np.argsort(dist_line)
    flags = np.zeros(len(gallery_name))
    result = {}
    for i in range(len(dist)):
        result[query_name[i].split(r'/')[-1]]=[]
    dist_cp = dist.copy()
    dist_cp.sort(1)
    dist_r1 = dist_cp[:,0]
    rank1 = np.argsort(dist_r1)
    dist_r1.sort()
    thr = dist_r1[int(len(rank1)*0.7)] # TODO
    print(thr)
    num = 0
    num_jump = 0
    for i in range(len(dist_sorted_indices)):
        index_query = dist_sorted_indices[i] // num_gallery
        index_gallery = dist_sorted_indices[i] % num_gallery
        d = dist_line[dist_sorted_indices[i]]
        #print(d)
        if d < thr:
            if flags[index_gallery] == 0:
                flags[index_gallery] = 1
                if len(result[query_name[index_query].split(r'/')[-1]]) < 200:
                    num += 1
                    result[query_name[index_query].split(r'/')[-1]].append(gallery_name[index_gallery].split(r'/')[-1])
            else:
                num_jump += 1
        else:
            if flags[index_gallery] == 0:
                if len(result[query_name[index_query].split(r'/')[-1]]) < 200:
                    num += 1
                    result[query_name[index_query].split(r'/')[-1]].append(gallery_name[index_gallery].split(r'/')[-1])
            else:
                num_jump += 1
        if num >= num_query*200:
            break
    print(num)
    print(num_jump)
    with open(r'submission_B.json', 'w', encoding='utf-8') as f:
        json.dump(result,f)
def write_json(dist, query_name, gallery_name):
    dist_cp = dist.copy()
    dist_cp.sort(1)
    dist_r1 = dist_cp[:, 0]
    rank1 = np.argsort(dist_r1)
    dist_r1.sort()
    flags = np.zeros(len(gallery_name))
    result = {}
    ttmmp = np.argsort(dist)
    thr = dist_r1[int(len(rank1) * 0.72)]
    for i in range(len(dist)):
        # if i%50 == 0:
        print(i)
        query_index = rank1[i]
        gallery_list = ttmmp[query_index]
        dist_i = dist[query_index]
        result[query_name[query_index].split(r'/')[-1]] = []
        num = 0
        first = True
        for g in gallery_list:
            if flags[g] == 1:
                first = False
                continue
            if first:
                flags[g] = 1
                first = False
            if dist_i[g] < thr:
                flags[g] = 1
            result[query_name[query_index].split(r'/')[-1]].append(gallery_name[g].split(r'/')[-1])
            num += 1
            if num == 200:
                break
    with open(r'submission_B.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)


def get_data():
    query_list = list()
    with open(r'/mnt/baseline/reid-baseline/origindata/初赛B榜测试集/query_b_list.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            img_file = os.path.join(r'/mnt/baseline/reid-baseline/origindata/初赛B榜测试集/query_b', image_name)
            query_list.append(img_file)
    gallery_list = [os.path.join(r'/mnt/baseline/reid-baseline/origindata/初赛B榜测试集/gallery_b', x) for x in
                    os.listdir(r'/mnt/baseline/reid-baseline/origindata/初赛B榜测试集/gallery_b')]
   
    #write_json(dist=distmat, query_name=query_list, gallery_name=gallery_list)
    write_json2(dist=distmat, query_name=query_list, gallery_name=gallery_list)

if __name__ == '__main__':
    get_data()
