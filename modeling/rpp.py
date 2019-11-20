import torch
import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.constant_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


class RPP(nn.Module):
    def __init__(self):
        super(RPP, self).__init__()
        self.part = 6
        add_block = []
        add_block += [nn.Conv2d(2048, 6, kernel_size=1, bias=False)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        norm_block = []
        norm_block += [nn.BatchNorm2d(2048)]
        norm_block += [nn.ReLU(inplace=True)]
        # norm_block += [nn.LeakyReLU(0.1, inplace=True)]
        norm_block = nn.Sequential(*norm_block)
        norm_block.apply(weights_init_kaiming)

        self.add_block = add_block
        self.norm_block = norm_block
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        w = self.add_block(x)
        p = self.softmax(w)
        y = []
        for i in range(self.part):
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.norm_block(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)

        f = torch.cat(y, 2)
        return f
