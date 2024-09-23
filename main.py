#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Saifullahi Aminu Bello
@Contact: saifullahiabel@szu.edu.cn
@File: main.py
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from GEConvNet import GEConvNet
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from ModelNet40Loader import ModelNet40Cls
from torchvision import transforms
import d_utils
from tqdm import tqdm

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp GEConvNet.py checkpoints' + '/' + args.exp_name + '/' + 'GEConvNet.py.backup')
    os.system('cp GEConvNet_util.py checkpoints' + '/' + args.exp_name + '/' + 'GEConvNet_util.py.backup')



def train(args, io):
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    train_dataset = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataset = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=test_transforms,
                                 train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    device = torch.device("cuda" if args.cuda else "cpu")

    model = GEConvNet(args).to(device)

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label, normal in train_loader:
            data, normal = d_utils.rotate_point_cloud_so3_norm(data, normal) # use d_utils.rotate_point_cloud_norm(data, normal) for Z rotation
            data, label, normal = (torch.from_numpy(data).to(device)).permute(0, 2, 1), label.to(device).squeeze(), (
                torch.from_numpy(normal).to(device)).permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data, normal)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        with torch.no_grad():
            model.eval()
            test_pred = []
            test_true = []
            for data, label, normal in test_loader:
                data, normal = d_utils.rotate_point_cloud_so3_norm(data, normal)
                data, label, normal = (torch.from_numpy(data).to(device)).permute(0, 2, 1), label.to(
                    device).squeeze(), (torch.from_numpy(normal).to(device)).permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data, normal)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  test_loss * 1.0 / count,
                                                                                  test_acc,
                                                                                  avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_dataset = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=test_transforms,
                                 train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4, #args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = GEConvNet(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    for data, label, normal in tqdm(test_loader):
        data, normal = d_utils.rotate_point_cloud_norm(data, normal)
        #rt = torch.rand(3).to(device)
        data, label, normal = (torch.from_numpy(data).to(device)).permute(0, 2, 1), label.to(device).squeeze(), (torch.from_numpy(normal).to(device)).permute(0, 2, 1)
        logits = model(data, normal)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--use_global', type=bool, default=False,
                        help='use global coordinate system')
    parser.add_argument('--with_normal', type=bool, default=True,
                        help='use surface normal for pdr')
    parser.add_argument('--model_path', type=str, default='pretrained/classification.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_root', type=str, default='data/',
                        metavar='data_root', help='data path')

    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
