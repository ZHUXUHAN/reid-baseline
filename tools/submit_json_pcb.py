# encoding:utf-8
# from dataset.data import read_image  # 图片读取方法，可以自己写，我是用的baseline里自带的
from data.datasets.dataset_loader import read_image
import os
import torch
import numpy as np
import json
# from evaluate import eval_func, euclidean_dist, re_rank #  计算距离以及rerank，均是来自baseline
import torchvision.transforms as T
from modeling.aligned_baseline import AlignedBaseline
from modeling.pcb_baseline import PCBBaseline
from utils.re_ranking import re_ranking
from utils.aligned_reranking import aligned_re_ranking
from torchvision import transforms
# from sklearn.decomposition import PCA
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    array1 = array1.cpu().numpy()
    array2 = array2.cpu().numpy()
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


def get_model(traindata_num_classes, modelname):
    pretrain_path = r"/home/zxh/.cache/torch/checkpoints/r50_ibn_a.pth"
    # model = AlignedBaseline(traindata_num_classes,1,pretrain_path,'bnneck','before','resnet50_ibn_a','self',gcb=True)
    model = PCBBaseline(traindata_num_classes, 1, pretrain_path, 'bnneck', 'after', 'resnet50_ibn_a', 'self', 6, False,
                        False, False, False, False, False)
    model.load_param(modelname)
    return model


def write_json(dist, query_name, gallery_name):
    rank = np.argsort(dist)
    rank0 = rank[:, 0]
    print(len(rank0))
    print(len(set(rank0)))
    dist_cp = dist.copy()
    dist_cp.sort(1)
    dist_r1 = dist_cp[:, 0]
    rank1 = np.argsort(dist_r1)
    dist_r1.sort()
    print(dist_r1)
    print(dist[rank1[1]][rank0[rank1[1]]])
    flags = np.zeros(len(gallery_name))
    result = {}
    thr = dist_r1[int(len(rank1) * 0.72)]
    ttppmm = np.argsort(dist)
    for i in range(len(dist)):
        if i % 50 == 0:
            print(i)
            print(sum(flags))
        query_index = rank1[i]
        gallery_list = ttppmm[query_index]
        dist_i = dist[query_index]
        result[query_name[query_index].split(r'/')[-1]] = []
        num = 0
        first = True
        for g in gallery_list:
            if flags[g] == 1:
                first = False
                continue
            if first:
                # if i < int(len(query_name)*0.85):
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


def inference_samples(model, batch_size):  # 传入模型，数据预处理方法，batch_size
    query_list = list()
    with open(r'/home/zxh/reid-baseline/query_b_list.txt', 'r') as f:
        # 测试集中txt文件
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            img_file = os.path.join(r'/home/zxh/naic_data/ori_data/初赛B榜测试集/query_b',
                                    image_name)  # 测试集query文件夹
            query_list.append(img_file)

    gallery_list = [os.path.join(r'/home/zxh/naic_data/ori_data/初赛B榜测试集/gallery_b', x) for x in
                    # 测试集gallery文件夹
                    os.listdir(r'/home/zxh/naic_data/ori_data/初赛B榜测试集/gallery_b')]
    query_num = len(query_list)
    # img_list = list()
    # flip_img_list = list()
    # for q_img in query_list:
    #     q_img = read_image(q_img)
    #     flip_q_img = transforms.RandomHorizontalFlip(p=1.0)(q_img)
    #     q_img = transform(q_img)
    #     flip_q_img = transform(flip_q_img)
    #     img_list.append(q_img)
    #     flip_img_list.append(flip_q_img)
    # for g_img in gallery_list:
    #     g_img = read_image(g_img)
    #     flip_g_img = transforms.RandomHorizontalFlip(p=1.0)(g_img)
    #     g_img = transform(g_img)
    #     flip_g_img = transform(flip_g_img)
    #     img_list.append(g_img)
    #     flip_img_list.append(flip_g_img)
    # img_data = torch.Tensor([t.numpy() for t in img_list])
    # flip_img_data = torch.Tensor([t.numpy() for t in flip_img_list])
    # img_data=img_data.to(device)
    # flip_img_data=flip_img_data.to(device)

    # model = nn.DataParallel(model, device_ids=device_ids)
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
    # pca = PCA(n_components=512)
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
            batch_data = batch_data.to(device)
            # print(batch_data.shape,"ooooooo")
            # batch_data.to(device)
            batch_feature = model(batch_data, None)[0].detach().cuda()
            batch_feature1 = model(batch_data, None)[1].detach().cuda()
            all_feature.append(batch_feature)
            local_feature.append(batch_feature1)

            flip_batch_data = flip_batch_data.to(device)
            flip_batch_feature = model(flip_batch_data, None)[0].detach().cuda()
            flip_batch_feature1 = model(flip_batch_data, None)[1].detach().cuda()
            flip_all_feature.append(flip_batch_feature)
            flip_local_feature.append(flip_batch_feature1)
        # flip_local_feature.append(flip_batch_feature1)
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
            # print(batch_data.shape,"ooooooo")
            # batch_data.to(device)
            batch_feature = model(batch_data, None)[0].detach().cuda()
            batch_feature1 = model(batch_data, None)[1].detach().cuda()
            all_feature.append(batch_feature)
            local_feature.append(batch_feature1)
            flip_batch_feature = model(flip_batch_data, None)[0].detach().cuda()
            flip_batch_feature1 = model(flip_batch_data, None)[1].detach().cuda()
            flip_all_feature.append(flip_batch_feature)
            flip_local_feature.append(flip_batch_feature1)

    all_feature = torch.cat(all_feature, dim=0)
    local_feature = torch.cat(local_feature, dim=0)
    flip_all_feature = torch.cat(flip_all_feature, dim=0)
    flip_local_feature = torch.cat(flip_local_feature, dim=0)
    # feat_norm == 'yes':
    all_feature = torch.nn.functional.normalize(all_feature, dim=1, p=2)
    local_feature = torch.nn.functional.normalize(local_feature, dim=1, p=2)
    flip_all_feature = torch.nn.functional.normalize(flip_all_feature, dim=1, p=2)
    flip_local_feature = torch.nn.functional.normalize(flip_local_feature, dim=1, p=2)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]
    flip_gallery_feat = flip_all_feature[query_num:]
    flip_query_feat = flip_all_feature[:query_num]

    local_gallery_feat = local_feature[query_num:]
    local_query_feat = local_feature[:query_num]
    flip_local_gallery_feat = flip_local_feature[query_num:]
    flip_local_query_feat = flip_local_feature[:query_num]

    # gallery_feat=pca.fit_transform(gallery_feat)
    # query_feat=pca.fit_transform(query_feat)
    # flip_gallery_feat=pca.fit_transform(flip_gallery_feat)
    # flip_query_feat=pca.fit_transform(flip_query_feat)

    # local_gallery_feat=pca.fit_transform(local_gallery_feat)
    # local_query_feat=pca.fit_transform(local_query_feat)
    # flip_local_gallery_feat=pca.fit_transform(flip_local_gallery_feat)
    # flip_local_query_feat=pca.fit_transform(flip_local_query_feat)

    global_q_g_dist = compute_dist(
        query_feat, gallery_feat, type='euclidean')
    global_g_g_dist = compute_dist(
        gallery_feat, gallery_feat, type='euclidean')
    global_q_q_dist = compute_dist(
        query_feat, query_feat, type='euclidean')
    flip_global_q_g_dist = compute_dist(
        flip_query_feat, flip_gallery_feat, type='euclidean')
    flip_global_g_g_dist = compute_dist(
        flip_gallery_feat, flip_gallery_feat, type='euclidean')
    flip_global_q_q_dist = compute_dist(
        flip_query_feat, flip_query_feat, type='euclidean')

    local_q_g_dist_all = []
    local_q_q_dist_all = []
    local_g_g_dist_all = []
    flip_local_q_g_dist_all = []
    flip_local_q_q_dist_all = []
    flip_local_g_g_dist_all = []
    # pcb_test or aligned_test:
    for i in range(local_query_feat.shape[2]):
        local_q_g_dist = compute_dist(
            local_query_feat[:, :, i], local_gallery_feat[:, :, i],
            type='euclidean')
        local_q_g_dist_all.append(local_q_g_dist)

        local_q_q_dist = compute_dist(
            local_query_feat[:, :, i], local_query_feat[:, :, i],
            type='euclidean')
        local_q_q_dist_all.append(local_q_q_dist)

        local_g_g_dist = compute_dist(
            local_gallery_feat[:, :, i], local_gallery_feat[:, :, i],
            type='euclidean')
        local_g_g_dist_all.append(local_g_g_dist)

        # ---------
        flip_local_q_g_dist = compute_dist(
            flip_local_query_feat[:, :, i], flip_local_gallery_feat[:, :, i],
            type='euclidean')
        flip_local_q_g_dist_all.append(flip_local_q_g_dist)

        flip_local_q_q_dist = compute_dist(
            flip_local_query_feat[:, :, i], flip_local_query_feat[:, :, i],
            type='euclidean')
        flip_local_q_q_dist_all.append(flip_local_q_q_dist)

        flip_local_g_g_dist = compute_dist(
            flip_local_gallery_feat[:, :, i], flip_local_gallery_feat[:, :, i],
            type='euclidean')
        flip_local_g_g_dist_all.append(flip_local_g_g_dist)

    global_local_g_g_dist = global_g_g_dist
    global_local_q_g_dist = global_q_g_dist
    global_local_q_q_dist = global_q_q_dist

    flip_global_local_g_g_dist = flip_global_g_g_dist
    flip_global_local_q_g_dist = flip_global_q_g_dist
    flip_global_local_q_q_dist = flip_global_q_q_dist

    for i in range(len(local_g_g_dist_all)):  # /len(local_g_g_dist_all)

        global_local_g_g_dist += local_g_g_dist_all[i] / (len(local_g_g_dist_all))
        global_local_q_g_dist += local_q_g_dist_all[i] / (len(local_g_g_dist_all))
        global_local_q_q_dist += local_q_q_dist_all[i] / (len(local_g_g_dist_all))

        flip_global_local_g_g_dist += flip_local_g_g_dist_all[i] / (len(flip_local_g_g_dist_all))
        flip_global_local_q_g_dist += flip_local_q_g_dist_all[i] / (len(flip_local_g_g_dist_all))
        flip_global_local_q_q_dist += flip_local_q_q_dist_all[i] / (len(flip_local_g_g_dist_all))

    global_local_q_g_dist += flip_global_local_q_g_dist
    global_local_q_q_dist += flip_global_local_q_q_dist
    global_local_g_g_dist += flip_global_local_g_g_dist
    # distmat = aligned_re_ranking(global_q_g_dist, global_q_q_dist, global_g_g_dist, k1=6, k2=3, lambda_value=0.80)
    distmat = aligned_re_ranking(global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist, k1=6, k2=3,
                                 lambda_value=0.85)
    # distmat = re_rank(query_feat, gallery_feat) # rerank方法
    # distmat=euclidean_dist(query_feat, gallery_feat)
    # distmat = distmat # 如果使用 euclidean_dist，不使用rerank改为：distamt = distamt.numpy()
    # distmat = distmat.numpy()
    np.save('distmat_82json_rank1_pcb_erase.npy', distmat)
    write_json(dist=distmat, query_name=query_list, gallery_name=gallery_list)


def main(traindata_num_classes, modelname, batch_size):
    model = get_model(traindata_num_classes, modelname)
    inference_samples(model, batch_size)


if __name__ == "__main__":
    traindata_num_classes = 9263
    modelname = "/home/zxh/reid-baseline/resnet50_ibn_a_model_80_82json_ran1_erase.pth"
    batch_size = 128
    main(traindata_num_classes, modelname, batch_size)
