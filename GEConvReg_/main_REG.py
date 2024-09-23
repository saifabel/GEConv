#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import ModelNet40
from util import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from GEConv_RANSAC import GEConv_ransac
from GEConvNet import GEConvNet
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def test_one_epoch(net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0


    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []


    eulers_ab = []


    for src, target, rotation_ab, translation_ab, _, t_, euler_ab, _, src_n, tgt_n in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        src_n=src_n.cuda()
        tgt_n=tgt_n.cuda()

        batch_size = src.size(0)
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = GEConv_ransac(net, src, src_n, target, tgt_n)

        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##


        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)

        total_loss += loss.item() * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size


    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)


    eulers_ab = np.concatenate(eulers_ab, axis=0)


    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred,  eulers_ab


def test(net, test_loader):

    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_eulers_ab = test_one_epoch(net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)


    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))


    print('==TEST==')

    print('rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))




def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--use_global', type=bool, default=False,
                        help='use global coordinate system, false for registration')
    parser.add_argument('--with_normal', type=bool, default=True,
                        help='use surface normal for pdr')
    parser.add_argument('--data_root', type=str, default='../data/',
                        metavar='data_root', help='data path')

    args = parser.parse_args()


    test_loader = DataLoader(
            ModelNet40(args.data_root, num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise, factor=args.factor),
            batch_size=3, shuffle=False, drop_last=False)

    net=GEConvNet(args).cuda()
    net = nn.DataParallel(net)
    if not args.unseen:
        net.load_state_dict(torch.load('model.t7'))
    net = net.eval()

    test(net, test_loader)


    print('FINISH')



if __name__ == '__main__':
    main()
