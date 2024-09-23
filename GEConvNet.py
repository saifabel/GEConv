#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from GEConvNet_util import GEConv, PointNetFeaturePropagation

class GEConvNet_partseg(nn.Module):
    def __init__(self, args, num_classes=50):
        super(GEConvNet_partseg, self).__init__()

        self.k = 40
        # Abstraction
        self.gec1 = GEConv(args, k=64, inp_dim=14, out_dim=64, npoint=None, dynm=True, layer1=True)
        self.gec2 = GEConv(args, k=64, inp_dim=64, out_dim=128, npoint=512, dynm=False)
        self.gec3 = GEConv(args, k=128, inp_dim= 128 , out_dim=128, npoint=None, dynm=False)
        self.gec4 = GEConv(args, k=64, inp_dim= 128 , out_dim=512, npoint= 64, dynm=False)
        self.gec5 = GEConv(args, k=8, inp_dim=512, out_dim=512, npoint=None, dynm=False)

        self.gbn = nn.BatchNorm1d(512)

        self.gconv = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.gbn,
                                   nn.LeakyReLU(negative_slope=0.2))

        #feature propagation
        self.fp5 = PointNetFeaturePropagation(in_channel=640+1024, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+64, mlp=[128, 128])

        # Classifier
        self.cconv1 = nn.Conv1d(128+16, 256, 1)
        self.cbn1 = nn.BatchNorm1d(256)
        self.cdrop1 = nn.Dropout(0.5)
        self.cconv2 = nn.Conv1d(256, num_classes, 1)

    def forward(self, x, cls_label, n=None):
        B, C, N = x.shape

        x, xyz_1, n_1 = self.gec1(x, x, n)
        x_1 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_2, n_2 = self.gec2(x_1, xyz_1, n_1)
        x_2 = x.max(dim=-1, keepdim=False)[0] #[B, C, N]

        x, xyz_3, n_3 = self.gec3(x_2, xyz_2, n_2)

        x_3 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_4, n_4 = self.gec4(x_3, xyz_3, n_3)

        x_4 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_5, n_5 = self.gec5(x_4, xyz_4, n_4)

        x_5 = x.max(dim=-1, keepdim=False)[0]
        x=self.gconv(x_5)
        gl_x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        gl_x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        gf=torch.cat((gl_x1, gl_x2), 1)

        x_5=torch.cat((x, gf.unsqueeze(-1).repeat(1, 1, x_5.size(-1))), 1)

        x_3 = self.fp5(xyz_3, xyz_5, x_3, x_5)
        x_1=self.fp1(xyz_1, xyz_3, x_1, x_3)

        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        x_1=torch.cat((x_1, cls_label_one_hot), 1)

        # Classifier
        feat = F.relu(self.cbn1(self.cconv1(x_1)))
        x = self.cdrop1(feat)
        x = self.cconv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x




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
        #self.conv2(x)
        x = x + resx1
        resx = x
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_3, n_3 = self.gec3(x2, xyz_2, n_2)
        #self.conv3(x)
        x = x + resx

        # resx3=x
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_4, n_4 = self.gec4(x3, xyz_3, n_3)
        # x=x+resx3
        x4 = x.max(dim=-1, keepdim=False)[0]


        x=self.conv4(torch.cat((x1, x2, x3, x4), 1))

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x



class GEConvNet_deep(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GEConvNet_deep, self).__init__()
        self.args = args
        k=args.k

        self.gec1 = GEConv(args, k=k, inp_dim=14, out_dim=32, npoint=None, dynm=True, layer1=True)
        self.gec2 = GEConv(args, k=k, inp_dim=32, out_dim=32, npoint=None, dynm=True)
        self.gec3 = GEConv(args, k=k, inp_dim=32, out_dim=48, npoint=None, dynm=True)
        self.gec4 = GEConv(args, k=k, inp_dim=48, out_dim=48, npoint=None, dynm=True)
        self.gec5 = GEConv(args, k=k, inp_dim=48, out_dim=64, npoint=None, dynm=True)
        self.gec6 = GEConv(args, k=k, inp_dim=64, out_dim=64, npoint=None, dynm=True)
        self.gec7 = GEConv(args, k=k, inp_dim=64, out_dim=128, npoint=None, dynm=True)
        self.gec8 = GEConv(args, k=k, inp_dim=128, out_dim=128, npoint=None, dynm=True)


        self.conv4 = nn.Sequential(nn.Conv1d(544, args.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
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

        x2 = x.max(dim=-1, keepdim=False)[0]
        x, xyz_3, n_3 = self.gec3(x2, xyz_2, n_2)

        resx3=x
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_4, n_4 = self.gec4(x3, xyz_3, n_3)
        x=x+resx3
        x4 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_5, n_5 = self.gec5(x4, xyz_4, n_4)
        resx5=x
        x5 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_6, n_6 = self.gec6(x5, xyz_5, n_5)
        x = x + resx5
        x6 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_7, n_7 = self.gec7(x6, xyz_6, n_6)
        resx7=x
        x7 = x.max(dim=-1, keepdim=False)[0]

        x, xyz_8, n_8 = self.gec8(x7, xyz_7, n_7)
        x = x + resx7
        x8 = x.max(dim=-1, keepdim=False)[0]

        x=self.conv4(torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 1))

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class GEConvNet_scan(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GEConvNet_scan, self).__init__()
        self.args = args
        k=args.k

        self.gec1 = GEConv(args, k=k, inp_dim=14, out_dim=int(64/4), npoint=None, dynm=True, layer1=True)
        self.gec2 = GEConv(args, k=k, inp_dim=int(64/4), out_dim=int(64/4), npoint=None, dynm=True)
        self.gec3 = GEConv(args, k=k, inp_dim=int(64/4), out_dim=int(64/4), npoint=None, dynm=True)
        self.gec4 = GEConv(args, k=k, inp_dim=int(64/4), out_dim=int(128), npoint=None, dynm=True)


        self.conv4 = nn.Sequential(nn.Conv1d(int(128), int(512/4), kernel_size=1, bias=False),
                                   nn.BatchNorm1d(int(512/4)),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(128, output_channels) #, bias=False)


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

        x = F.adaptive_avg_pool1d(x4, 1).view(batch_size, -1)

        return  F.log_softmax(self.linear1(x), dim=1)