# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler
import json


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        _pids = []
        _pids_100 = []
        with open('./data/samplers/number.txt', 'r') as f:
            _pids_list = eval(f.readline().strip())
        for i in range(len(_pids_list)):
            _pids.append(int(_pids_list[i]))

        with open('./data/samplers/number_more100.txt', 'r') as f:
            _pids_100_list = eval(f.readline().strip())
        for i in range(len(_pids_100_list)):
            _pids_100.append(int(_pids_100_list[i]))

        with open('./data/samplers/data.json', 'r') as f:
            self.data = json.load(f)

        self._pids_100 = _pids_100
        self._pids = _pids

        for index, (_, pid, _) in enumerate(self.data_source):
            if pid in self._pids:
                self.index_dic[pid].append(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            # elif len(idxs) > self.num_instances and len(idxs) % self.num_instances != 0:
            #     idxs = np.random.choice(idxs, size=len(idxs) + self.num_instances - (len(idxs) % self.num_instances),
            #                             replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)  # 每个元素列表中 里面只有4个id
                    batch_idxs = []

        final_idxs = []

        if len(self.pids) % self.num_pids_per_batch != 0:
            avai_pids = np.random.choice(self.pids,
                                         size=len(self.pids) + self.pids - (len(self.pids) % self.num_pids_per_batch),
                                         replace=True)
        else:
            avai_pids = copy.deepcopy(self.pids)  # 最终是从这个地方出来的

        while len(avai_pids) >= self.num_pids_per_batch:  # 只要大于等于就会遍历
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)  # 16
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        # max = 0
        # avai_id = 0
        # for i, id in enumerate(avai_pids):
        #     if len(self.data[str(id)]) > max:
        #         max = len(self.data[str(id)])
        #         avai_id = id
        # avai_pids.append(avai_id)
        # batch_idxs_dict_copy = copy.deepcopy(batch_idxs_dict[avai_id])
        # assert len(avai_pids) == self.num_pids_per_batch, 'Error'
        #
        # while len(avai_pids) > 0:  # 只要大于等于就会遍历
        #     for i, pid in enumerate(avai_pids):
        #         if i == len(avai_pids)-1:
        #             batch_idxs_residue_copy = batch_idxs_dict_copy.pop(0)
        #             assert len(batch_idxs_residue_copy) == 4, 'Error'
        #             final_idxs.extend(batch_idxs_residue_copy)
        #         elif pid != avai_id:
        #             batch_idxs_residue = batch_idxs_dict[pid].pop(0)
        #             assert len(batch_idxs_residue) == 4, 'Error'
        #             final_idxs.extend(batch_idxs_residue)
        #         elif pid == avai_id and i != len(avai_pids)-1:
        #             batch_idxs_residue = batch_idxs_dict[avai_id].pop(0)
        #             assert len(batch_idxs_residue) == 4, 'Error'
        #             final_idxs.extend(batch_idxs_residue)
        #
        #         if len(batch_idxs_dict[pid]) == 0 and pid != avai_id:
        #             avai_pids.remove(pid)
        #         elif len(batch_idxs_dict[avai_id]) == 0 and i != len(avai_pids)-1:
        #             del(avai_pids[i])
        #         elif len(batch_idxs_dict_copy) == 0:
        #             del(avai_pids[-1])

        print("One Epoch's Batches For Train Size Are", len(final_idxs))

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
