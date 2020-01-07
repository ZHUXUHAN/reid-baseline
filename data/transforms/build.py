# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, RandomResize
from .augmix import AugMix


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    PIXEL_MEAN = (0.212737, 0.1830550, 0.097212)

    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # RandomResize(384, 128),
            # T.RandomAffine(degrees=0, translate=(0.05, 0.05), fillcolor=PIXEL_MEAN),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            # AugMix(prob=0.5),
            RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=PIXEL_MEAN),

        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
