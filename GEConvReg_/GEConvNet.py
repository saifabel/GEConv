#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch.nn as nn
#sys.path.append('../GEConvNet_master')
from GEConvNet_util import GEConv

class GEConvNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GEConvNet, self).__init__()
        self.args = args
        k=args.k

        self.gec1 = GEConv(args, k=k, inp_dim=14, out_dim=64, npoint=None, dynm=True, layer1=True)
        self.gec2 = GEConv(args, k=k, inp_dim=64, out_dim=64, npoint=None, dynm=True)
        self.gec3 = GEConv(args, k=k, inp_dim=64, out_dim=64, npoint=None, dynm=True)
        self.gec4 = GEConv(args, k=k, inp_dim=64, out_dim=128, npoint=None, dynm=True)


        self.conv4 = nn.Sequential(nn.Conv1d(64*3+128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, n=None):
        batch_size = x.size(0)

        x, xyz_1, n_1 = self.gec1(x, x, n)

        resx1 = x
        x1 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_2, n_2 = self.gec2(x1, xyz_1, n_1)
        x = x + resx1
        resx = x
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_3, n_3 = self.gec3(x2, xyz_2, n_2)
        x = x + resx

        x3 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_4, n_4 = self.gec4(x3, xyz_3, n_3)
        x4 = x.max(dim=-1, keepdim=False)[0]

        return x1.permute(0, 2, 1), xyz_1



