import sys

import torch
import numpy as np
from GEConvNet import GEConvNet
from ransac.utils import ransac_pose_estimation as ransac_pose
from ransac.metrics import compute_metrics as cmetrics

def saif_visualize_result(template, source, est_T, k):
	template, source, est_T = template, source, est_T
	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]

	np.savetxt('D:/codes/Registration/vis2/GEConvNet/transformed_'+str(k)+'_.txt', transformed_source, delimiter=' ')

def GEConv_ransac(model, src, src_n, tgt, tgt_n):
    ##############################################
    '''k = [5, 8, 10, 24, 26, 27, 33, 35, 39]
    all_src = []
    all_tgt = []
    for i in range(len(k)):
        src = np.loadtxt('D:/codes/Registration/vis2/src_' + str(k[i]) + '_.txt')
        tgt = np.loadtxt('D:/codes/Registration/vis2/tgt_' + str(k[i]) + '_.txt')

        all_src.append(np.expand_dims(src, 0))
        all_tgt.append(np.expand_dims(tgt, 0))
    tgt = torch.from_numpy(np.concatenate(all_tgt, 0).astype('float32'))[:, :, :3].permute(0,2,1)
    tgt_n = torch.from_numpy(np.concatenate(all_tgt, 0).astype('float32'))[:, :, 3:].permute(0,2,1)
    src = torch.from_numpy(np.concatenate(all_src, 0).astype('float32'))[:, :, :3].permute(0,2,1)
    src_n = torch.from_numpy(np.concatenate(all_src, 0).astype('float32'))[:, :, 3:].permute(0,2,1)'''
    #################################################
    src_feat, src=model(src, src_n)
    tgt_feat, tgt=model(tgt, tgt_n)
    est = []




    src=src.permute(0,2,1) ######
    tgt=tgt.permute(0,2,1)#########
    for i in range(src_feat.size()[0]):
        est_i = ransac_pose(src[i].detach().cpu().numpy(), tgt[i].detach().cpu().numpy(),
                            src_feat[i].detach().cpu().numpy(), tgt_feat[i].detach().cpu().numpy())  # s, t, s_f, t_f, mutual = False, distance_threshold=0.05, ransac_n=3

        #saif_visualize_result(tgt[i].detach().cpu().numpy(), src[i].detach().cpu().numpy(), est_i, k[i])
        est.append(est_i)
    #sys.exit()
    est = np.stack(est).astype('float32')


    R = torch.from_numpy(est[:, :3, :3]).cuda() #.transpose(0, 2, 1)
    T = torch.from_numpy(est[:, :3, 3]).cuda()
    #bb=torch.cat((R, T), 2)
    #print(torch.from_numpy((est)==bb))
    #print(est.shape, bb.shape)
    #sys.exit()
    return R, T, R, T