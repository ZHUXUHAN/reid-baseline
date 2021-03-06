# encoding:utf-8
# from dataset.data import read_image  # 图片读取方法，可以自己写，我是用的baseline里自带的
from data.datasets.dataset_loader import read_image
import os
import torch
import numpy as np
import json
from data.datasets.eval_reid import eval_func
import torchvision.transforms as T
from modeling.aligned_baseline import AlignedBaseline
from modeling.pcb_baseline import PCBBaseline
from utils.re_ranking import re_ranking
from utils.aligned_reranking import aligned_re_ranking
from torchvision import transforms
# from sklearn.decomposition import PCA
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2,6'  # 指定gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import cfg
from data import make_data_loader
import re


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
      array1: numpy array with shape [m1, n]
      array2: numpy array with shape [m2, n]
      type: one of ['cosine', 'euclidean']
    Returns:
      numpy array with shape [m1, m2]
    """
    # array1 = array1.cpu().numpy()
    # array2 = array2.cpu().numpy()
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def re_rank(qf, gf):
    return re_ranking(qf, gf, k1=6, k2=2, lambda_value=0.4)


def transform(img):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = T.Compose([
        T.Resize([384, 128]),
        T.ToTensor(),
        normalize_transform
    ])
    result = result(img)
    return result


def compute_qg_dist(all_feature, query_num):
    gf = all_feature[query_num:]
    qf = all_feature[:query_num]

    q_g_dist = compute_dist(
        qf.cpu().detach().numpy(), gf.cpu().detach().numpy(), type='euclidean')
    g_g_dist = compute_dist(
        gf.cpu().detach().numpy(), gf.cpu().detach().numpy(), type='euclidean')
    q_q_dist = compute_dist(
        qf.cpu().detach().numpy(), qf.cpu().detach().numpy(), type='euclidean')
    return q_g_dist, q_q_dist, g_g_dist


def get_model(traindata_num_classes, modelname):
    pretrain_path = r"/home/zxh/reid-baseline/new_experiment/r101_pcb_new/resnext101_ibn_a_model_70.pth"
    model = PCBBaseline(traindata_num_classes, 1, pretrain_path, 'bnneck', 'after', 'resnext101_ibn_a', 'self', 6, False,
                        False, False, False, False, False)
    model.load_param(modelname)
    return model


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
        qimg_path, qpid, qcamid = query_name[i]
        qimg_name = self.parse_filename(osp.basename(qimg_path))
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
        qimg_path, qpid, qcamid = query_name[index_query]
        qimg_name = self.parse_filename(osp.basename(qimg_path))
        gimg_path, gpid, gcamid = gallery_name[index_gallery]
        gimg_name = self.parse_filename(osp.basename(gimg_path))
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


def adjust_rerank_function(dist_list, query_pid_list, gallery_pid_list, query_camid_list, gallery_camid_list):
    q_pids = np.asarray(query_pid_list)
    g_pids = np.asarray(gallery_pid_list)
    q_camids = np.asarray(query_camid_list)
    g_camids = np.asarray(gallery_camid_list)

    max = 0
    plist = []
    global_q_g_dist, global_q_q_dist, global_g_g_dist = dist_list[0:3]
    local_q_g_dist, local_q_q_dist, local_g_g_dist = dist_list[3:6]
    flip_global_q_g_dist, flip_global_q_q_dist, flip_global_g_g_dist = dist_list[6:9]
    flip_local_q_g_dist, flip_local_q_q_dist, flip_local_g_g_dist = dist_list[9:12]

    for k1 in range(6, 8, 1):
        for k2 in range(3, 5, 1):
            for l in [0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]:  #
                for l_w in [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]:

                    distmat_global = aligned_re_ranking(
                        global_q_g_dist, global_q_q_dist, global_g_g_dist, k1=k1, k2=k2,
                        lambda_value=l)
                    print("Global dismat is computed done")

                    distmat_local = aligned_re_ranking(
                        local_q_g_dist, local_q_q_dist, local_g_g_dist, k1=k1, k2=k2,
                        lambda_value=l)

                    print("Local dismat is computed done")

                    flip_distmat_global = aligned_re_ranking(
                        flip_global_q_g_dist, flip_global_q_q_dist, flip_global_g_g_dist, k1=k1, k2=k2,
                        lambda_value=l)

                    print("Global_flip dismat is computed done")

                    flip_distmat_local = aligned_re_ranking(
                        flip_local_q_g_dist, flip_local_q_q_dist, flip_local_g_g_dist, k1=k1, k2=k2,
                        lambda_value=l)

                    print("Local_flip dismat is computed done")

                    distmat = l_w * distmat_global + (1 - l_w) * distmat_local

                    flip_distmat = l_w * flip_distmat_global + (1 - l_w) * flip_distmat_local

                    distmat = (flip_distmat + distmat) / 2

                    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                    for r in [1]:
                        if max < (mAP + cmc[r - 1]) / 2:
                            max = (mAP + cmc[r - 1]) / 2
                            plist = [k1, k2, l, mAP, cmc[r - 1]]
                        print("====k1=%d=====k2=%d=====l=%f=====l_w=%f" % (k1, k2, l, l_w))
                        print("CMC curve, Rank-%d:%.4f, map:%.4f, final: %.4f" % (
                            r, cmc[r - 1], mAP, (mAP + cmc[r - 1]) / 2))

    print(plist)


def inference_samples(model, batch_size, query_list, gallery_list, query_pid_list, gallery_pid_list, query_camid_list, \
                          gallery_camid_list, adjust_rerank, npy_save_path):  # 传入模型，数据预处理方法，batch_size
    query_num = len(query_list)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    iter_n_query = (len(query_list)) // batch_size
    iter_n_gallery = (len(gallery_list)) // batch_size
    if (len(query_list)) % batch_size != 0:
        iter_n_query += 1
    if (len(gallery_list)) % batch_size != 0:
        iter_n_gallery += 1
    all_feature = list()
    local_feature = list()
    flip_all_feature = list()
    flip_local_feature = list()
    print("ALL_QUERY_BACTH:", str(iter_n_query))
    for i in range(iter_n_query):
        img_list = list()
        flip_img_list = list()
        for q_img in query_list[i * batch_size:(i + 1) * batch_size]:
            q_img = read_image(q_img)
            flip_q_img = transforms.RandomHorizontalFlip(p=1.0)(q_img)
            q_img = transform(q_img)
            flip_q_img = transform(flip_q_img)
            img_list.append(q_img)
            flip_img_list.append(flip_q_img)
        img_data = torch.Tensor([t.numpy() for t in img_list]).to(device)
        flip_img_data = torch.Tensor([t.numpy() for t in flip_img_list]).to(device)
        print("batch ----%d----" % (i))
        batch_data = img_data
        flip_batch_data = flip_img_data
        with torch.no_grad():
            batch_feature = model(batch_data, None)[0].detach().cuda()
            local_batch_feature = model(batch_data, None)[1].detach().cuda()
            all_feature.append(batch_feature)
            local_feature.append(local_batch_feature)

            flip_batch_feature = model(flip_batch_data, None)[0].detach().cuda()
            local_flip_batch_feature = model(flip_batch_data, None)[1].detach().cuda()
            flip_all_feature.append(flip_batch_feature)
            flip_local_feature.append(local_flip_batch_feature)
    print("ALL_GALLERY_BACTH:", str(iter_n_gallery))
    for i in range(iter_n_gallery):
        img_list = list()
        flip_img_list = list()
        for g_img in gallery_list[i * batch_size:(i + 1) * batch_size]:
            g_img = read_image(g_img)
            flip_g_img = transforms.RandomHorizontalFlip(p=1.0)(g_img)
            g_img = transform(g_img)
            flip_g_img = transform(flip_g_img)
            img_list.append(g_img)
            flip_img_list.append(flip_g_img)
        img_data = torch.Tensor([t.numpy() for t in img_list]).to(device)
        flip_img_data = torch.Tensor([t.numpy() for t in flip_img_list]).to(device)
        batch_data = img_data
        flip_batch_data = flip_img_data
        with torch.no_grad():
            print("batch ----%d----" % (i))
            batch_feature = model(batch_data, None)[0].detach().cuda()
            local_batch_feature = model(batch_data, None)[1].detach().cuda()
            all_feature.append(batch_feature)
            local_feature.append(local_batch_feature)

            flip_batch_feature = model(flip_batch_data, None)[0].detach().cuda()
            local_flip_batch_feature = model(flip_batch_data, None)[1].detach().cuda()
            flip_all_feature.append(flip_batch_feature)
            flip_local_feature.append(local_flip_batch_feature)

    all_feature = torch.cat(all_feature, dim=0)
    local_feature = torch.cat(local_feature, dim=0)
    flip_all_feature = torch.cat(flip_all_feature, dim=0)
    flip_local_feature = torch.cat(flip_local_feature, dim=0)

    #Flip Cat
    # all_feature = torch.cat((all_feature, flip_all_feature), dim=1)
    # local_feature = torch.cat((local_feature, flip_local_feature), dim=1)
    ######
    all_feature = torch.nn.functional.normalize(all_feature, dim=1, p=2)
    local_feature = torch.nn.functional.normalize(local_feature, dim=1, p=2)
    flip_all_feature = torch.nn.functional.normalize(flip_all_feature, dim=1, p=2)
    flip_local_feature = torch.nn.functional.normalize(flip_local_feature, dim=1, p=2)

    global_q_g_dist, global_g_g_dist, global_q_q_dist = compute_qg_dist(all_feature, query_num)
    local_q_g_dist, local_g_g_dist, local_q_q_dist = compute_qg_dist(local_feature, query_num)
    flip_global_q_g_dist, flip_global_g_g_dist, flip_global_q_q_dist = compute_qg_dist(flip_all_feature, query_num)
    flip_local_q_g_dist, flip_local_g_g_dist, flip_local_q_q_dist = compute_qg_dist(flip_local_feature, query_num)

    if adjust_rerank:
        dist_list = [global_q_g_dist, global_g_g_dist, global_q_q_dist, local_q_g_dist, local_g_g_dist, local_q_q_dist, \
                      flip_global_q_g_dist, flip_global_g_g_dist, flip_global_q_q_dist, flip_local_q_g_dist,
                      flip_local_g_g_dist, flip_local_q_q_dist]
        adjust_rerank_function(dist_list, query_pid_list, gallery_pid_list, query_camid_list, gallery_camid_list)
    else:
        l_w = 0.95
        k1, k2, l = 6, 3, 0.80
        distmat_global = aligned_re_ranking(
            global_q_g_dist, global_q_q_dist, global_g_g_dist, k1=k1, k2=k2,
            lambda_value=l)
        print("Global dismat is computed done")

        distmat_local = aligned_re_ranking(
            local_q_g_dist, local_q_q_dist, local_g_g_dist, k1=k1, k2=k2,
            lambda_value=l)

        print("Local dismat is computed done")

        flip_distmat_global = aligned_re_ranking(
            flip_global_q_g_dist, flip_global_q_q_dist, flip_global_g_g_dist, k1=k1, k2=k2,
            lambda_value=l)

        print("Global_flip dismat is computed done")

        flip_distmat_local = aligned_re_ranking(
            flip_local_q_g_dist, flip_local_q_q_dist, flip_local_g_g_dist, k1=k1, k2=k2,
            lambda_value=l)

        print("Local_flip dismat is computed done")

        distmat = l_w * distmat_global + (1 - l_w) * distmat_local

        flip_distmat = l_w * flip_distmat_global + (1 - l_w) * flip_distmat_local

        distmat = (flip_distmat + distmat) / 2
        np.save(npy_save_path, distmat)


def merge_npy(npy_paths):
    '''
    需要自动化调整这个函数的写法
    :param npy_paths: 包含所有npypath的list
    :return:
    '''
    npy_dist_1 = np.load(npy_paths[0])
    npy_dist_2 = np.load(npy_paths[1])
    distmat = npy_dist_1+npy_dist_2
    write_json(dist=distmat, query_name=query_list, gallery_name=gallery_list, save_dir='./', topk=200)


def main(traindata_num_classes, modelname, batch_size, merge_npy_paths, adjust_rerank , npy_save_path):
    if len(merge_npy_paths) > 0:
        merge_npy(merge_npy_paths)
    else:
        query_list = [os.path.join(r'/home/zxh/datasets/NAICReID/clean_query', x) for x in
                      # 测试集gallery文件夹
                      os.listdir(r'/home/zxh/datasets/NAICReID/clean_query')]

        gallery_list = [os.path.join(r'/home/zxh/datasets/NAICReID/clean_bounding_box_test', x) for x in
                        # 测试集gallery文件夹
                        os.listdir(r'/home/zxh/datasets/NAICReID/clean_bounding_box_test')]
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

        model = get_model(traindata_num_classes, modelname)
        inference_samples(model, batch_size, query_list, gallery_list, query_pid_list, gallery_pid_list, query_camid_list, \
                          gallery_camid_list, adjust_rerank, npy_save_path)


if __name__ == "__main__":
    """
    需要更换的有 modelname, adjust_rerank 如果不是调整rerank 则需要给出需要融合的 npy_save_path
    """
    traindata_num_classes = 9968
    modelname = "/home/zxh/reid-baseline/new_experiment/r101_pcb_new/resnext101_ibn_a_model_70.pth"
    batch_size = 128*2
    merge_npy_paths = []
    adjust_rerank = True
    npy_save_path = ''
    main(traindata_num_classes, modelname, batch_size, merge_npy_paths, adjust_rerank, npy_save_path)
