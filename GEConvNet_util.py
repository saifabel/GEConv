import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

pdist = nn.PairwiseDistance(p=2)
cos = torch.nn.CosineSimilarity(dim=3)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # centroids[:, i] = farthest
        if i == 0:
            centroid = torch.mean(xyz, 1).view(B, 1, 3)  # xyz[batch_indices, farthest, :].view(B, 1, 3)
        else:
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        centroids[:, i] = farthest
    return centroids


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
def PDR(x, new_x, norms=None, nnorms=None, use_global=False):
    B, S, K, _ = x.shape
    nx = x - new_x.unsqueeze(2)
    mean_x = torch.mean(nx, 2).unsqueeze(2)
    pdist1 = torch.norm(nx, dim=3).unsqueeze(1)  ############ f2
    pdist2 = torch.norm(nx - mean_x, dim=3).unsqueeze(1)  ########## f3
    cos1 = cos(nx, mean_x).unsqueeze(1)  ############# f5

    if norms==None:
        mean_x2 = torch.mean(x, 2).unsqueeze(2)
        norms=x-mean_x2
        nnorms=new_x.unsqueeze(2)-mean_x2
        pdist3 = torch.norm((nx + norms) - (nnorms), dim=3).unsqueeze(1)
        cos2 = cos(nx, norms).unsqueeze(1)
        cos3 = cos(norms, nnorms.view(B, S, 1, 3)).unsqueeze(1)

        if use_global==True:
            pdist4 = torch.norm(x, dim=3).unsqueeze(1)
            cos4 = cos(x, new_x.unsqueeze(2)).unsqueeze(1)
            cos5 = cos(x, norms).unsqueeze(1)
            output = torch.cat((pdist1, pdist2,  pdist3, pdist4, cos5, cos4, cos1, cos2, cos3), 1)

        else:
            output = torch.cat((pdist1, pdist2, pdist3, cos1, cos2, cos3), 1)



    else:

        mean_x2 = torch.mean(x, 2).unsqueeze(2)
        nxnorms = x - mean_x2
        nnxnorms = new_x.unsqueeze(2) - mean_x2
        norms = norms - new_x.unsqueeze(2)
        nnorms = nnorms.unsqueeze(2) - new_x.unsqueeze(2)
        pdist3 = torch.norm(nx - nnorms, dim=3).unsqueeze(1)
        pdist4 = torch.norm((nx + norms) - (nnorms), dim=3).unsqueeze(1)
        pdist5 = torch.norm((nx + nxnorms) - (nnorms), dim=3).unsqueeze(1)
        cos2 = cos(nx, norms).unsqueeze(1)
        cos3 = cos(norms, nnorms.view(B, S, 1, 3)).unsqueeze(1)
        cos4 = cos(nx, nxnorms).unsqueeze(1)
        cos5 = cos(nxnorms, nnxnorms.view(B, S, 1, 3)).unsqueeze(1)
        if use_global==True:
            pdist0 = torch.norm(x, dim=3).unsqueeze(1)
            cos6 = cos(x, new_x.unsqueeze(2)).unsqueeze(1)
            cos7 = cos(x, norms).unsqueeze(1)
            output = torch.cat((pdist0,  pdist1, pdist2, pdist3, pdist4, pdist5, cos1, cos2, cos3, cos4, cos5, cos7, cos6),1)
        else:
            output = torch.cat((pdist1, pdist2, pdist3, pdist4, pdist5, cos1, cos2, cos3, cos4, cos5),1)


    return torch.cat((output, output - torch.mean(output, -1).unsqueeze(-1)), 1)


def get_graph_feature_invr(f, xyz, n, nnum_points, k=20, idx=None, dynm=True, layer1=False, use_global=False):
    batch_size = f.size(0)
    num_points = f.size(2)

    f = f.view(batch_size, -1, num_points)

    nnp = nnum_points
    if idx is None:
        if dynm == True:
            idx = knn(f, k=k)  # f: features, xyz: coordinates. dynamic or static respectively
        else:
            idx = knn(xyz, k=k)
    device = torch.device('cuda')

    if nnum_points is not None:
        f_ps = farthest_point_sample(xyz.permute(0, 2, 1), nnum_points)
        idx = index_points(idx, f_ps)
        nxyz = index_points(xyz.permute(0, 2, 1), f_ps)  # .permute(0,2,1)
        nn=None
        if n is not None:
            nn = index_points(n.permute(0, 2, 1), f_ps)  # .permute(0,2,1)
    else:
        nnum_points = num_points

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = xyz.size()
    xyz = xyz.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x_knn = xyz.view(batch_size * num_points, -1)[idx, :]
    x_knn = x_knn.view(batch_size, nnum_points, k, num_dims)

    if n is not None:
        n = n.transpose(2, 1).contiguous()
        n_knn = n.view(batch_size * num_points, -1)[idx, :]
        n_knn = n_knn.view(batch_size, nnum_points, k, num_dims)
    else: n_knn=None
    if nnp is not None:
        xyz = nxyz
        n = nn

    pdx = PDR(x_knn, xyz, n_knn, n, use_global)  # 10, 1024, 96, 3],[10, 1024, 3]
    if n is not None:
        n = n.permute(0, 2, 1)

    if layer1 is False:

        f = f.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = f.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, nnum_points, k, -1)
        if nnp is not None:
            f = index_points(f, f_ps).permute(0, 2, 1).contiguous()
        f = f.view(batch_size, nnum_points, 1, -1).repeat(1, 1, k, 1)

        feature = torch.cat((feature - f, f), dim=3).permute(0, 3, 1, 2).contiguous()

        return torch.cat((pdx, feature), 1), xyz.permute(0, 2, 1), n

    else:

        return pdx, xyz.permute(0, 2, 1), n


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class GEConv(nn.Module):
    def __init__(self,  args, k, inp_dim, out_dim, npoint=None, dynm=True, layer1=False):
        super(GEConv, self).__init__()
        self.with_normal = args.with_normal
        self.use_global=args.use_global
        self.npoint = npoint
        self.layer = layer1
        self.k = k
        self.dynm = dynm

        if args.with_normal:
            geomet_edge_dim = 20
        else:
            geomet_edge_dim = 12
        if args.use_global:
            geomet_edge_dim+=6
        if not layer1:
            geomet_edge_dim+=(inp_dim*2)

        self.conv = nn.Sequential(nn.Conv2d(geomet_edge_dim, out_dim, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(out_dim),
                                          nn.GELU())



        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.GELU())

    def forward(self, x, xyz, norm=None):
        if self.with_normal is False:
            norm=None
        x, xyz, n = get_graph_feature_invr(x, xyz, norm, self.npoint, self.k, None, self.dynm, self.layer, self.use_global)

        x = self.conv2(self.conv(x))
        #x = self.conv2(x)

        return x, xyz, n



