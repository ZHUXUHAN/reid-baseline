# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking


def create_supervised_evaluator(model, aligned_test, pcb_test, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, data_flip = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            data_flip = data_flip.to(device) if torch.cuda.device_count() >= 1 else data_flip
            if aligned_test or pcb_test:
                feat, local_feat = model(data, None)
                flip_feat, flip_local_feat = model(data_flip, None)
                return feat, local_feat, pids, camids, flip_feat, flip_local_feat
            else:
                feat = model(data)
                flip_feat = model(data_flip)
                return feat, pids, camids, flip_feat

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query,
        datasets
):
    device = cfg.MODEL.DEVICE
    aligned_test = cfg.MODEL.ALIGNED
    adjust_rerank = cfg.TEST.ADJUST_RERANK
    pcb_test = cfg.MODEL.PCB
    ggdist_path = cfg.TEST.SAVE_DIST_GG
    qqdist_path = cfg.TEST.SAVE_DIST_QQ
    qgdist_path = cfg.TEST.SAVE_DIST_QG
    savedist_path = [ggdist_path, qqdist_path, qgdist_path]
    merge = cfg.TEST.MERGE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, aligned_test, pcb_test, metrics={'r1_mAP': R1_mAP(num_query, datasets, aligned_test, pcb_test, adjust_rerank, savedist_path, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, aligned_test, pcb_test, metrics={'r1_mAP': R1_mAP_reranking(num_query, datasets, aligned_test, pcb_test, adjust_rerank, savedist_path, merge, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    # cmc, mAP = evaluator.state.metrics['r1_mAP']
    # logger.info('Validation Results')
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
