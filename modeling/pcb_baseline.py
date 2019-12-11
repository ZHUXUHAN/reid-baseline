# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import os
import copy
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.components.attention import PAM_Module, CAM_Module
# from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, Bottleneck_IBN, IBN
from .backbones.senet_ibn_a import se_resnet101_ibn_a, SEBottleneck
from .rpp import RPP
from .arcface_loss import ArcCos
from .backbones.densenet import densenet121, densenet169, densenet201, densenet161
from .backbones.inception import inceptionv4
from .patchgenerator import PatchGenerator
from .batchdrop import BatchDrop
from .classblock import ClassBlock
from .stn import STN


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class PCBBaseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, num_stripes,
                 gcb, rpp, dropout, cam, sum, arc):
        super(PCBBaseline, self).__init__()
        self.num_stripes = num_stripes
        self.gcb = gcb
        self.rpp = rpp
        self.dropout = dropout
        self.hidden_dim = 256
        self.cam = cam
        self.sum = sum
        self.arc = arc
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride, self.gcb, self.cam)

        elif model_name == 'resnet101_ibn_a':
            self.base = resnet101_ibn_a(last_stride, self.gcb)

        elif model_name == 'se_resnet101_ibn_a':
            self.base = se_resnet101_ibn_a(last_stride)

        elif model_name == 'inception':
            self.base = InceptionNet(last_stride)

        if pretrain_choice == 'imagenet':
            if not os.path.exists(model_path):
               assert "No The Pretrained Model"
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        # #PatchGenerator
        self.patch_proposal = PatchGenerator()
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.batch_crop = BatchDrop(1, 0.05)

        self.base.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        self.base.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))

        # self.stn = STN()

        if self.sum:
            print("Building Attention Branch")
            self.atten_pam = PAM_Module(256)
            self.atten_cam = CAM_Module(256)
            sum_conv = nn.Sequential(
                nn.Dropout2d(0.1, False),
                nn.Conv2d(256, 256, kernel_size=1)
            )
            sum_conv.apply(weights_init_kaiming)
            self.sum_conv = sum_conv

        # Add new layers
        if self.rpp:
            self.avgpool = RPP()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((self.num_stripes, 1))

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=0.5)

        if self.arc:
            self.arcface = ArcCos(self.in_planes, self.num_classes, s=30.0, m=0.50)

        self.local_conv_list = nn.ModuleList()

        for _ in range(self.num_stripes):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, self.hidden_dim, kernel_size=1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True))
            local_conv.apply(weights_init_kaiming)
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe
        self.fc_list = nn.ModuleList()

        if self.neck == 'no':
            # For Global
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)  # new add by luo
            # For Part
            for _ in range(self.num_stripes):
                fc = nn.Linear(self.hidden_dim, self.num_classes)
                fc.apply(weights_init_classifier)
                self.fc_list.append(fc)
        elif self.neck == 'bnneck':
            self.bottleneck_list = nn.ModuleList()
            # For Global
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            for _ in range(self.num_stripes):
                fc = nn.Linear(self.hidden_dim, self.num_classes, bias=False)
                fc.apply(weights_init_classifier)
                self.fc_list.append(fc)

                bottleneck = nn.BatchNorm1d(self.hidden_dim)
                bottleneck.bias.requires_grad_(False)  # no shift
                bottleneck.apply(weights_init_kaiming)

                self.bottleneck_list.append(bottleneck)

    def forward(self, x, label):
        # x = self.stn(x)  #### stn
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        resnet_features = self.base(x)
        resnet_features_patch = self.patch_proposal(resnet_features)

        # [N, C, H, W]
        assert resnet_features.size(
            2) % self.num_stripes == 0, 'Image height cannot be divided by num_strides, and the feature shape is {}'.format(resnet_features.shape)
        features_G = self.avgpool(resnet_features)
        if self.dropout and self.training:
            features_G = self.dropout_layer(features_G)  # dropout only used in training
        # [N, C=256, H=S, W=1]
        features_H = []
        for i in range(self.num_stripes):
            stripe_features_H = F.adaptive_avg_pool2d(resnet_features_patch[i], (1, 1)).squeeze(-1)
            # print(features_G[:, :, i, :].shape)
            stripe_features_H_conv = self.local_conv_list[i][0](
                stripe_features_H)#
            local_stripe_features_H = self.local_conv_list[i][1:](
                stripe_features_H_conv.unsqueeze(-1))
            if self.sum:
                stripe_features_H_pam = self.atten_pam(local_stripe_features_H)
                stripe_features_H_cam = self.atten_cam(local_stripe_features_H)
                stripe_features_H_sum = self.sum_conv(sum(stripe_features_H_pam, stripe_features_H_cam))
            if self.neck == 'no':
                stripe_features_H = stripe_features_H
            elif self.neck == 'bnneck':
                if self.sum:
                    stripe_features_H = self.bottleneck_list[i](
                        stripe_features_H_sum.squeeze(-1))  # normalize for angular softmax
                else:
                    stripe_features_H = self.bottleneck_list[i](
                        local_stripe_features_H.squeeze(-1))  # normalize for angular softmax
            if self.training:
                features_H.append(local_stripe_features_H.squeeze())
            else:
                features_H.append(local_stripe_features_H.squeeze())

        if self.training:
            # [N, C=num_classes]
            batch_size = x.size(0)
            logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1))
                           for i in range(self.num_stripes)]
            features_H = torch.stack(features_H, dim=2)
            features_H = features_H.view(features_H.size(0), -1)

            if self.arc:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            # global_score global_ft part_score part_ft
            return cls_score, global_feat, logits_list, features_H
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN") # torch.stack(features_H, dim=2)
                return feat, torch.cat((features_H[0], features_H[1], features_H[2], features_H[3], features_H[4], \
                                        features_H[5]), 1)
            else:
                # print("Test with feature before BN")
                return global_feat, torch.stack(features_H, dim=2)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
