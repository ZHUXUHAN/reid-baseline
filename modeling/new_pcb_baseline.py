# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import os
import copy

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.components.attention import PAM_Module, CAM_Module
# from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, Bottleneck_IBN, IBN
from .backbones.senet_ibn_a import se_resnet101_ibn_a, SEBottleneck
from .rpp import RPP
from .arcface_loss import ArcCos
from .backbones.densenet import densenet121, densenet169, densenet201, densenet161


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


class NEW_PCBBaseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, num_stripes,
                 gcb, rpp, dropout, cam, sum, arc):
        super(NEW_PCBBaseline, self).__init__()
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

        self.backbone = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3[0]
        )
        res_conv4 = nn.Sequential(*self.base.layer3[1:])
        res_g_conv5 = self.base.layer4  # global 2倍下采样

        base_copy = copy.deepcopy(self.base)

        base_copy.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        base_copy.layer4[0].conv2.apply(weights_init_kaiming)
        base_copy.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))
        base_copy.layer4[0].downsample.apply(weights_init_kaiming)

        self.res_conv4_local = copy.deepcopy(base_copy.layer4)
        self.res_conv4_local_2 = copy.deepcopy(base_copy.layer4)
        # res_p_conv5 = nn.Sequential(
        #     Bottleneck_IBN(1024, 512, ibn=False,
        #                    downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
        #     Bottleneck_IBN(2048, 512, ibn=False),
        #     Bottleneck_IBN(2048, 512, ibn=False))
        # res_p_conv5.load_state_dict(self.base.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))  # global
        # self.p2 = nn.Sequential(copy.deepcopy(res_conv4_local), copy.deepcopy(res_p_conv5))  # part
        # self.p3 = nn.Sequential(copy.deepcopy(res_conv4_local), copy.deepcopy(res_p_conv5))  # part

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
            self.avgpool_2 = nn.AdaptiveAvgPool2d((self.num_stripes, 1))

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=0.5)

        if self.arc:
            self.arcface = ArcCos(self.in_planes, self.num_classes, s=30.0, m=0.50)

        self.local_conv_list = nn.ModuleList()
        self.local_2_conv_list = nn.ModuleList()

        for _ in range(self.num_stripes):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, self.hidden_dim, kernel_size=1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True))
            local_conv.apply(weights_init_kaiming)
            self.local_conv_list.append(local_conv)

        for _ in range(3):
            local_2_conv = nn.Sequential(
                nn.Conv1d(2048, self.hidden_dim, kernel_size=1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True))
            local_2_conv.apply(weights_init_kaiming)
            self.local_2_conv_list.append(local_2_conv)

        # Classifier for each stripe
        self.fc_list = nn.ModuleList()
        self.fc_2_list = nn.ModuleList()

        if self.neck == 'no':
            # For Global
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)  # new add by luo
            # For Part
            for _ in range(self.num_stripes):
                fc = nn.Linear(self.hidden_dim, self.num_classes)
                fc.apply(weights_init_classifier)
                self.fc_list.append(fc)
            for _ in range(3):
                fc_2 = nn.Linear(self.hidden_dim, self.num_classes)
                fc_2.apply(weights_init_classifier)
                self.fc_2_list.append(fc_2)
        elif self.neck == 'bnneck':
            self.bottleneck_list = nn.ModuleList()
            self.bottleneck_2_list = nn.ModuleList()
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
            for _ in range(3):
                fc_2 = nn.Linear(self.hidden_dim, self.num_classes, bias=False)
                fc_2.apply(weights_init_classifier)
                self.fc_2_list.append(fc_2)

                bottleneck_2 = nn.BatchNorm1d(self.hidden_dim)
                bottleneck_2.bias.requires_grad_(False)  # no shift
                bottleneck_2.apply(weights_init_kaiming)

                self.bottleneck_2_list.append(bottleneck)

    def forward(self, x, label):
        # global_feat = self.gap(self.p1(self.backbone(x)))  # (b, 2048, 1, 1)
        global_feat = self.gap(self.base(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        resnet_features = self.backbone(x)

        # [N, C, H, W]
        assert resnet_features.size(
            2) % self.num_stripes == 0, 'Image height cannot be divided by num_strides'
        features_G = self.avgpool(self.res_conv4_local(resnet_features))
        features_G_2 = self.avgpool_2(self.res_conv4_local_2(resnet_features))
        if self.dropout and self.training:
            features_G = self.dropout_layer(features_G)  # dropout only used in training
            features_G_2 = self.dropout_layer(features_G_2)  # dropout only used in training
        # [N, C=256, H=S, W=1]
        features_H = []
        features_H_2 = []
        for i in range(self.num_stripes):
            stripe_features_H_conv = self.local_conv_list[i][0](
                features_G[:, :, i, :])
            stripe_features_H = self.local_conv_list[i][1:](
                stripe_features_H_conv.unsqueeze(-1))
            if self.sum:
                stripe_features_H_pam = self.atten_pam(stripe_features_H)
                stripe_features_H_cam = self.atten_cam(stripe_features_H)
                stripe_features_H_sum = self.sum_conv(sum(stripe_features_H_pam, stripe_features_H_cam))
            if self.neck == 'no':
                stripe_features_H = stripe_features_H
            elif self.neck == 'bnneck':
                if self.sum:
                    stripe_features_H = self.bottleneck_list[i](
                        stripe_features_H_sum.squeeze(-1))  # normalize for angular softmax
                else:
                    stripe_features_H = self.bottleneck_list[i](
                        stripe_features_H.squeeze(-1))  # normalize for angular softmax
            features_H.append(stripe_features_H.squeeze())
        for i in range(3):
            stripe_features_H_2_conv = self.local_2_conv_list[i][0](
                features_G_2[:, :, i, :])
            stripe_features_H_2 = self.local_conv_list[i][1:](
                stripe_features_H_2_conv.unsqueeze(-1))
            if self.sum:
                stripe_features_H_pam = self.atten_pam(stripe_features_H_2)
                stripe_features_H_cam = self.atten_cam(stripe_features_H_2)
                stripe_features_H_sum = self.sum_conv(sum(stripe_features_H_pam, stripe_features_H_cam))
            if self.neck == 'no':
                stripe_features_H_2 = stripe_features_H_2
            elif self.neck == 'bnneck':
                if self.sum:
                    stripe_features_H_2 = self.bottleneck_list[i](
                        stripe_features_H_sum.squeeze(-1))  # normalize for angular softmax
                else:
                    stripe_features_H_2 = self.bottleneck_list[i](
                        stripe_features_H_2.squeeze(-1))  # normalize for angular softmax
            features_H_2.append(stripe_features_H_2.squeeze())

        if self.training:
            # [N, C=num_classes]
            batch_size = x.size(0)
            logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1))
                           for i in range(self.num_stripes)]
            logits_list_2 = [self.fc_2_list[i](features_H_2[i].view(batch_size, -1))
                           for i in range(3)]
            if self.arc:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            # global_score global_ft part_score part_ft
            return cls_score, global_feat, logits_list, features_H, logits_list_2, features_H_2
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat, torch.stack(features_H, dim=2), torch.stack(features_H_2, dim=2)
            else:
                # print("Test with feature before BN")
                return global_feat, torch.stack(features_H, dim=2), torch.stack(features_H_2, dim=2)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])