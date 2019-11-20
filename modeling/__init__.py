# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .aligned_baseline import AlignedBaseline
from .pcb_baseline import PCBBaseline
from .mgn_baseline import MGNBaseline


def build_model(cfg, num_classes):
    if cfg.MODEL.ALIGNED:
        model = AlignedBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                                cfg.TEST.NECK_FEAT,
                                cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.WITH_GCB)
    elif cfg.MODEL.PCB:
        model = PCBBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                                cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.PCB_NUM_STRIPES,
                            cfg.MODEL.WITH_GCB, cfg.MODEL.PCB_RPP, cfg.MODEL.DROPOUT, cfg.MODEL.CAM, cfg.MODEL.SUM, cfg.MODEL.ARC)
    elif cfg.MODEL.MGN:
        model = MGNBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                            cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.WITH_GCB)

    else:
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT,
                         cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.WITH_GCB)
    return model
