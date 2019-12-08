# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .local_loss import local_loss
from .local_triplet import TripletLoss_Local

import torch


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, local_score, local_feat):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    global_loss = triplet(feat, target)[0]
                    center_loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                    global_score_loss = xent(score, target)
                    tripletloss_local = TripletLoss_Local(0.3)
                    if cfg.MODEL.ALIGNED:
                        pinds, ginds = triplet(feat, target)[3], triplet(feat, target)[4]
                        return xent(score, target) + \
                               global_loss + center_loss + \
                               local_loss(tripletloss_local, local_feat, pinds, ginds, target)[0]
                    elif cfg.MODEL.PCB:
                        return xent(score, target) + \
                               global_loss + \
                               sum(xent(s, target) for s in local_score) / len(local_score) + \
                               sum(triplet(f, target)[0] for f in local_feat) / len(local_feat)

                    else:
                        return global_score_loss + global_loss + center_loss
                else:
                    if cfg.MODEL.ALIGNED:
                        return F.cross_entropy(score, target) + triplet(feat, target)[0]
                    else:
                        return sum(F.cross_entropy(s, target) for s in score) / len(score) + sum(
                            triplet(f, target)[0] for f in feat) / len(feat)

            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):  # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    elif cfg.MODEL.NAME == 'densenet161':
        feat_dim = 2208
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        center_criterion_local = CenterLoss(num_classes=num_classes, feat_dim=5120, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_focal=cfg.MODEL.USE_FOCAL_LOSS)  # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target, local_score, local_feat, local_score_2, local_feat_2):  # local_feat
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if cfg.MODEL.ALIGNED:
                    tripletloss_local = TripletLoss_Local(0.3)
                    pinds, ginds = triplet(feat, target)[3], triplet(feat, target)[4]
                    global_loss = triplet(feat, target)[0]
                    return xent(score, target) + \
                           global_loss + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                           local_loss(tripletloss_local, local_feat, pinds, ginds, target)[0]
                elif cfg.MODEL.PCB:
                    # tripletloss_local = TripletLoss_Local(0.3)
                    # local_metric_loss = triplet(local_feat, target)[0]
                    global_loss = triplet(feat, target)[0]
                    # pinds, ginds = triplet(feat, target)[3], triplet(feat, target)[4]
                    return cfg.SOLVER.GLOBAL_IDLOSS_WEIGHT*xent(score, target) + \
                           cfg.SOLVER.GLOBAL_METRICLOSS_WEIGHT*global_loss + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                           cfg.SOLVER.LOCAL_IDLOSS_WEIGHT * sum(xent(s, target) for s in local_score) / len(local_score)
                           # local_loss(tripletloss_local, local_feat.unsqueeze(-1), pinds, ginds, target)[0]
                           # local_metric_loss

                elif cfg.MODEL.NEW_PCB:#local_metric_loss +
                    global_metric_loss = triplet(feat, target)[0]
                    # local_metric_loss = triplet(local_feat, target)[0]
                    global_id_loss = xent(score, target)
                    local_id_loss = sum(xent(s, target) for s in local_score) / len(local_score)
                    return global_id_loss + \
                           global_metric_loss +  \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                           local_id_loss
                           # cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_local(local_feat, target)
                    # sum(xent(s, target) for s in local_score_2) / len(local_score_2)
                elif cfg.MODEL.MGN:
                    return sum(F.cross_entropy(s, target) for s in score) / len(score) + \
                           sum(triplet(f, target)[0] for f in feat) / len(feat)
                    # cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(local_feat, target)
                    # cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

                    # sum(xent(s, target) for s in score) / len(score) + \

                else:
                    global_loss = triplet(feat, target)[0]
                    return xent(score, target) + \
                           global_loss + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

            else:
                return F.cross_entropy(score, target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    return loss_func, center_criterion
