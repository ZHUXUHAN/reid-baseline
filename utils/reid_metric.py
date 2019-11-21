# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric
import os.path as osp
import os
import json
import sys

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from .aligned_reranking import aligned_re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, aligned_test, datasets, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.datasets = datasets
        self.aligned_test = aligned_test
        self.adjust_rerank = adjust_rerank
        self.merge = True

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def parse_filename(self, filename):
        resname = filename.split('_')[2] + '.png'
        return resname

    def write_json_results(self, distmat, dataset, save_dir='', topk=10):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        num_q, num_g = distmat.shape

        print('# query: {}\n# gallery {}'.format(num_q, num_g))
        print('Writing top-{} ranks ...'.format(topk))

        query, gallery = dataset.query, dataset.gallery
        assert num_q == len(query)
        assert num_g == len(gallery)

        indices = np.argsort(distmat, axis=1)
        json_dict = {}

        for q_idx in range(num_q):
            qimg_path, qpid, qcamid = query[q_idx]
            qimg_name = self.parse_filename(osp.basename(qimg_path))
            json_dict[qimg_name] = []

            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                gimg_path, gpid, gcamid = gallery[g_idx]
                gimg_nmae = osp.basename(gimg_path)
                if self.parse_filename(gimg_nmae) in json_dict[qimg_name]:
                    continue
                else:
                    json_dict[qimg_name].append(self.parse_filename(gimg_nmae))

                rank_idx += 1
                if rank_idx > topk:
                    break

            if (q_idx + 1) % 100 == 0:
                print('- done {}/{}'.format(q_idx + 1, num_q))

        with open(osp.join(save_dir, 'submission.json'), 'w', encoding='utf-8') as f:
            json.dump(json_dict, f)

        print('Done. Json have been saved to "{}" ...'.format(save_dir))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        self.write_json_results(
            distmat,
            self.datasets,
            save_dir=osp.join('/home/zxh/ReID/reid-strong-baseline/new_experiment/json_output', 'writerank_nrtireid'),
            topk=200
        )
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, datasets, aligned_test, pcb_test, adjust_rerank, savedist_path, merge, max_rank=50,
                 feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.datasets = datasets
        self.aligned_test = aligned_test
        self.pcb_test = pcb_test
        self.adjust_rerank = adjust_rerank
        self.savedist_gg = savedist_path[0]
        self.savedist_qq = savedist_path[1]
        self.savedist_qg = savedist_path[2]
        self.merge = merge

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.local_feats = []
        self.flip_feat = []
        self.flip_local_feat = []

    def update(self, output):
        if self.aligned_test or self.pcb_test:
            feat, local_feat, pid, camid, flip_feat, flip_local_feat = output
            feat = (feat+flip_feat)/2
            local_feat = (local_feat+flip_local_feat)/2
            self.feats.append(feat)
            self.local_feats.append(local_feat)
            self.flip_feat.append(flip_feat)
            self.flip_local_feat.append(flip_local_feat)
        else:
            feat, pid, camid, flip_feat = output
            self.feats.append(feat)
            self.flip_feat.append(flip_feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def parse_filename(self, filename):
        resname = filename.split('_')[2] + '.png'
        return resname

    def normalize(self, nparray, order=2, axis=0):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def write_json_results(self, distmat, dataset, save_dir='', topk=10, cat_num=0):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        num_q, num_g = distmat.shape

        print(distmat.shape)

        print('# query: {}\n# gallery {}'.format(num_q, num_g))
        print('Writing top-{} ranks ...'.format(topk))

        query, gallery = dataset.query, dataset.gallery

        for cat in range(cat_num):
            gallery += gallery

        # assert num_q == len(query)
        # assert num_g == len(gallery)

        indices = np.argsort(distmat, axis=1)
        json_dict = {}

        for q_idx in range(num_q):
            qimg_path, qpid, qcamid = query[q_idx]
            qimg_name = self.parse_filename(osp.basename(qimg_path))
            json_dict[qimg_name] = []

            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                gimg_path, gpid, gcamid = gallery[g_idx]
                gimg_nmae = osp.basename(gimg_path)
                if self.parse_filename(gimg_nmae) in json_dict[qimg_name]:
                    continue
                else:
                    json_dict[qimg_name].append(self.parse_filename(gimg_nmae))

                rank_idx += 1
                if rank_idx > topk:
                    break

            if (q_idx + 1) % 100 == 0:
                print('- done {}/{}'.format(q_idx + 1, num_q))

        with open(osp.join(save_dir, 'submission.json'), 'w', encoding='utf-8') as f:
            json.dump(json_dict, f)

        print('Done. Json have been saved to "{}" ...'.format(save_dir))

    def compute_dist(self, array1, array2, type='euclidean'):
        """Compute the euclidean or cosine distance of all pairs.
        Args:
          array1: numpy array with shape [m1, n]
          array2: numpy array with shape [m2, n]
          type: one of ['cosine', 'euclidean']
        Returns:
          numpy array with shape [m1, m2]
        """
        assert type in ['cosine', 'euclidean']
        if type == 'cosine':
            array1 = self.normalize(array1, axis=1)
            array2 = self.normalize(array2, axis=1)
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

    def compute(self):

        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if not self.merge:
            feats = torch.cat(self.feats, dim=0)
            flip_feat = torch.cat(self.flip_feat, dim=0)
            if self.aligned_test or self.pcb_test:
                local_feats = torch.cat(self.local_feats, dim=0)
                flip_local_feat = torch.cat(self.flip_local_feat, dim=0)

            if self.feat_norm == 'yes':
                print("The test feature is normalized")
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
                flip_feats = torch.nn.functional.normalize(flip_feat, dim=1, p=2)
                if self.aligned_test or self.pcb_test:
                    local_feats = torch.nn.functional.normalize(local_feats, dim=1, p=2)
                    flip_local_feats = torch.nn.functional.normalize(flip_local_feat, dim=1, p=2)

            if self.aligned_test or self.pcb_test:

                qf = feats[:self.num_query]
                gf = feats[self.num_query:]
                local_qf = local_feats[:self.num_query]
                local_gf = local_feats[self.num_query:]
                flip_qf = flip_feats[:self.num_query]
                flip_gf = flip_feats[self.num_query:]
                if self.aligned_test or self.pcb_test:
                    flip_local_qf = flip_local_feats[:self.num_query]
                    flip_local_gf = flip_local_feats[self.num_query:]

                global_q_g_dist = self.compute_dist(
                    qf.cpu().detach().numpy(), gf.cpu().detach().numpy(), type='euclidean')
                global_g_g_dist = self.compute_dist(
                    gf.cpu().detach().numpy(), gf.cpu().detach().numpy(), type='euclidean')
                global_q_q_dist = self.compute_dist(
                    qf.cpu().detach().numpy(), qf.cpu().detach().numpy(), type='euclidean')

                flip_global_q_g_dist = self.compute_dist(
                    flip_qf.cpu().detach().numpy(), flip_gf.cpu().detach().numpy(), type='euclidean')
                flip_global_g_g_dist = self.compute_dist(
                    flip_gf.cpu().detach().numpy(), flip_gf.cpu().detach().numpy(), type='euclidean')
                flip_global_q_q_dist = self.compute_dist(
                    flip_qf.cpu().detach().numpy(), flip_qf.cpu().detach().numpy(), type='euclidean')

                local_q_g_dist_all = []
                local_q_q_dist_all = []
                local_g_g_dist_all = []

                flip_local_q_g_dist_all = []
                flip_local_q_q_dist_all = []
                flip_local_g_g_dist_all = []

                if self.pcb_test or self.aligned_test:
                    for i in range(local_qf.shape[2]):
                        local_q_g_dist = self.compute_dist(
                            local_qf[:, :, i].cpu().detach().numpy(), local_gf[:, :, i].cpu().detach().numpy(),
                            type='euclidean')  # 1061,2233
                        local_q_g_dist_all.append(local_q_g_dist)

                        local_q_q_dist = self.compute_dist(
                            local_qf[:, :, i].cpu().detach().numpy(), local_qf[:, :, i].cpu().detach().numpy(),
                            type='euclidean')
                        local_q_q_dist_all.append(local_q_q_dist)

                        local_g_g_dist = self.compute_dist(
                            local_gf[:, :, i].cpu().detach().numpy(), local_gf[:, :, i].cpu().detach().numpy(),
                            type='euclidean')
                        local_g_g_dist_all.append(local_g_g_dist)

                        #####FLIP####
                        # flip_local_q_g_dist = self.compute_dist(
                        #     flip_local_qf[:, :, i].cpu().detach().numpy(),
                        #     flip_local_gf[:, :, i].cpu().detach().numpy(),
                        #     type='euclidean')
                        # flip_local_q_g_dist_all.append(flip_local_q_g_dist)
                        #
                        # flip_local_q_q_dist = self.compute_dist(
                        #     flip_local_qf[:, :, i].cpu().detach().numpy(),
                        #     flip_local_qf[:, :, i].cpu().detach().numpy(),
                        #     type='euclidean')
                        # flip_local_q_q_dist_all.append(flip_local_q_q_dist)
                        #
                        # flip_local_g_g_dist = self.compute_dist(flip_local_gf[:, :, i].cpu().detach().numpy(),
                        #                                         flip_local_gf[:, :, i].cpu().detach().numpy(),
                        #                                         type='euclidean')
                        # flip_local_g_g_dist_all.append(flip_local_g_g_dist)

                global_local_g_g_dist = global_g_g_dist
                global_local_q_g_dist = global_q_g_dist
                global_local_q_q_dist = global_q_q_dist

                for i in range(len(local_g_g_dist_all)):  # /len(local_g_g_dist_all)range(len(local_g_g_dist_all))

                    global_local_g_g_dist += local_g_g_dist_all[i]/len(local_g_g_dist_all)
                    global_local_q_g_dist += local_q_g_dist_all[i]/len(local_g_g_dist_all)
                    global_local_q_q_dist += local_q_q_dist_all[i]/len(local_g_g_dist_all)

            else:
                qf = feats[:self.num_query]
                gf = feats[self.num_query:]

            print("Enter reranking")

            if self.adjust_rerank:
                max = 0
                plist = []
                for k1 in range(6, 8, 1):
                    for k2 in range(3, 5, 1):
                        for l in [0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]:  #
                            if self.aligned_test or self.pcb_test:
                                distmat = aligned_re_ranking(
                                    global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist, k1=k1, k2=k2,
                                    lambda_value=l)
                                cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                                for r in [1]:
                                    if max < (mAP + cmc[r - 1]) / 2:
                                        max = (mAP + cmc[r - 1]) / 2
                                        plist = [k1, k2, l]
                                    print("====k1=%d=====k2=%d=====l=%f" % (k1, k2, l))
                                    print("CMC curve, Rank-%d:%.4f, map:%.4f, final: %.4f" % (
                                        r, cmc[r - 1], mAP, (mAP + cmc[r - 1]) / 2))
                            else:
                                distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=l)

                                cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                                for r in [1]:
                                    if max < (mAP + cmc[r - 1]) / 2:
                                        max = (mAP + cmc[r - 1]) / 2
                                        plist = [k1, k2, l]
                                    print("====k1=%d=====k2=%d=====l=%f" % (k1, k2, l))
                                    print("CMC curve, Rank-%d:%.4f, map:%.4f, final: %.4f" % (
                                        r, cmc[r - 1], mAP, (mAP + cmc[r - 1]) / 2))
                print(max, plist)
            else:
                if self.aligned_test or self.pcb_test:
                    distmat = aligned_re_ranking(
                        global_q_g_dist, global_q_q_dist, global_g_g_dist, k1=6, k2=4, lambda_value=0.78)

                else:
                    distmat = re_ranking(qf, gf, k1=6, k2=3, lambda_value=0.80)

                path_dist = os.path.join('./model_dist/global_local', 'erase0.3')
                if not os.path.exists(path_dist):
                    os.makedirs(path_dist)
                np.save(os.path.join(path_dist, 'dist.npy'), distmat)
                print("Save Npy Done")
        else:
            print('Entering Concated')
            distmat_1 = np.load('./model_dist/global_local/sampler/dist.npy')
            distmat_2 = np.load('./test_dist/nofc/dist.npy')
            # distmat = np.load('/home/zxh/ReID/reid-strong-baseline/model_dist/local/erase0.3/dist.npy')
            distmat = np.hstack((distmat_1, distmat_2))
            print("Dismat Concated Done")
            print("Entering Write Json File")
            self.write_json_results(
                distmat,
                self.datasets,
                save_dir=osp.join('/home/zxh/ReID/reid-strong-baseline/new_experiment/json_output',
                                  'writerank_nrtireid'),
                topk=200,
                cat_num=2
            )

        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        #
        # return cmc, mAP
