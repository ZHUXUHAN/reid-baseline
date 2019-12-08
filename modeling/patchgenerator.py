import torch.nn as nn
import torch.nn.functional as F
import torch


class PatchGenerator(nn.Module):
    def __init__(self):
        super(PatchGenerator, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 2 * 3 * 6),
        )

        path_postion = [1, 0, 0, 0, 1 / 6, -5 / 6,
                        1, 0, 0, 0, 1 / 6, -3 / 6,
                        1, 0, 0, 0, 1 / 6, -1 / 6,
                        1, 0, 0, 0, 1 / 6, 1 / 6,
                        1, 0, 0, 0, 1 / 6, 3 / 6,
                        1, 0, 0, 0, 1 / 6, 5 / 6, ]

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1, 1))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 6, 2, 3)

        output = []
        for i in range(6):
            stripe = theta[:, i, :, :]
            grid = F.affine_grid(stripe, x.size())
            output.append(F.grid_sample(x, grid))

        return output
