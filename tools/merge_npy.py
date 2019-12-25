import numpy as np
import os.path as osp
import json
import os
import glob
import re
from tqdm import tqdm


def parse_filename(filename):
    resname = filename.split('_')[2] + '.png'
    return resname


def write_json(dist, query_name, gallery_name, save_dir='', topk=10):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    num_q, num_g = dist.shape

    print(dist.shape)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Writing top-{} ranks ...'.format(topk))

    num_query, num_gallery = dist.shape
    print(dist.shape)
    dist_line = dist.reshape(-1)
    print(dist_line.shape)
    dist_sorted_indices = np.argsort(dist_line)
    flags = np.zeros(len(gallery_name))
    result = {}
    for i in range(len(dist)):
        qimg_path = query_name[i]
        qimg_name = parse_filename(osp.basename(qimg_path))
        result[qimg_name] = []
    dist_cp = dist.copy()
    dist_cp.sort(1)
    dist_r1 = dist_cp[:, 0]
    rank1 = np.argsort(dist_r1)
    dist_r1.sort()
    thr = dist_r1[int(len(rank1) * 0.72)]  # TODO
    print(thr)
    num = 0
    num_jump = 0
    for i in tqdm(range(len(dist_sorted_indices))):
        index_query = dist_sorted_indices[i] // num_gallery
        index_gallery = dist_sorted_indices[i] % num_gallery
        d = dist_line[dist_sorted_indices[i]]
        qimg_path = query_name[index_query]
        qimg_name = parse_filename(osp.basename(qimg_path))
        gimg_path = gallery_name[index_gallery]
        gimg_name = parse_filename(osp.basename(gimg_path))
        if d < thr:
            if flags[index_gallery] == 0:
                flags[index_gallery] = 1
                if len(result[qimg_name]) < topk:
                    num += 1
                    result[qimg_name].append(gimg_name)
            else:
                num_jump += 1
        else:
            if flags[index_gallery] == 0:
                if len(result[qimg_name]) < topk:
                    num += 1
                    result[qimg_name].append(gimg_name)
            else:
                num_jump += 1
        if num >= num_query * topk:
            break
    print(num)
    print(num_jump)
    with open(os.path.join(save_dir, r'submission_second_a.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f)
    print("Json Save Done")


def merge_npy(npy_paths):
    '''
    需要自动化调整这个函数的写法
    :param npy_paths: 包含所有npypath的list
    :return:
    '''
    npy_dist_1 = np.load(npy_paths[0])
    npy_dist_2 = np.load(npy_paths[1])
    npy_dist_3 = np.load(npy_paths[2])
    distmat = 0.7 * npy_dist_1 + 0.2 * npy_dist_2 + 0.1 * npy_dist_3

    query_list = [os.path.join(r'/home/zxh/datasets/NAICReID/query', x) for x in
                  # 测试集gallery文件夹
                  glob.glob(osp.join(r'/home/zxh/datasets/NAICReID/query', '*.png'))]

    gallery_list = [os.path.join(r'/home/zxh/datasets/NAICReID/clean_bounding_box_test', x) for x in
                    # 测试集gallery文件夹
                    glob.glob(osp.join(r'/home/zxh/datasets/NAICReID/bounding_box_test', '*.png'))]
    pattern = re.compile(r'([-\d]+)_c(\d)')

    query_pid_list = []
    query_camid_list = []
    for img_path in query_list:
        pid, camid = map(int, pattern.search(img_path).groups())
        query_pid_list.append(pid)
        query_camid_list.append(camid)

    gallery_camid_list = []
    gallery_pid_list = []
    for img_path in gallery_list:
        pid, camid = map(int, pattern.search(img_path).groups())
        gallery_pid_list.append(pid)
        gallery_camid_list.append(camid)

    num_query, num_gallery = distmat.shape
    assert len(query_list) == num_query
    assert len(gallery_list) == num_gallery

    write_json(dist=distmat, query_name=query_list, gallery_name=gallery_list, save_dir='./', topk=200)


merge_npy(['/home/zxh/distmat_75_TO_ZXH.npy', '/home/zxh/reid-baseline/model_dist/dist/101/dist.npy',
           '/home/zxh/reid-baseline/model_dist/dist/data/dist.npy'])
