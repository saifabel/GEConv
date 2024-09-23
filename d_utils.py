import torch
import numpy as np
import torch.nn as nn
import open3d as o3d

def compute_normal(np_array):
    pcd = o3d.geometry.PointCloud()
    # pcd = o3d.utility.Vector3dVector(np.load('...')[:, :3])
    pcd.points = o3d.utility.Vector3dVector(np_array)
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=200))
    pcd.orient_normals_to_align_with_direction()
    fp = np.asarray(pcd.points)
    fn = np.asarray(pcd.normals)
    #fpn = np.concatenate((fp, fn), 1)
    return fn #fpn  # np.savetxt('processed.txt', fpn, delimiter=' ')
def compute_norm(batch_data):
    
    normal = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):

        shape_pc = batch_data[k, ...]
        normal[k, ...] = compute_normal(shape_pc.reshape(-1, 3))
        
    return torch.from_numpy(normal)

def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)




class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

def angle_axis(angle: float, axis: np.ndarray):
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()    

class PointcloudRotatebyAngle(object):
    def __init__(self, rotation_angle = 0.0):
        self.rotation_angle =np.random.uniform() * 2 * np.pi #rotation_angle

    def __call__(self, pc):
        normals = pc.size(2) > 3
        bsize = pc.size()[0]
        for i in range(bsize):
            cosval = np.cos(self.rotation_angle)
            sinval = np.sin(self.rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            rotation_matrix = torch.from_numpy(rotation_matrix).float().cuda()
            
            cur_pc = pc[i, :, :]
            if not normals:
                cur_pc = cur_pc @ rotation_matrix
            else:
                pc_xyz = cur_pc[:, 0:3]
                pc_normals = cur_pc[:, 3:]
                cur_pc[:, 0:3] = pc_xyz @ rotation_matrix
                cur_pc[:, 3:] = pc_normals @ rotation_matrix
                
            pc[i, :, :] = cur_pc

            
        return pc, rotation_matrix

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc
        
class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc
        
class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc




def rotate_point_cloud_so3_norm(batch_data, norm_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape along arbitrary direction
        Input:
          BxNx3 array, original batch of point clouds
          BxNx3 array, original batch normals
        Return:
          BxNx3 array, rotated batch of point clouds
          BxNx3 array, rotated batch of normals
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_norm = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_A = np.random.uniform() * 2 * np.pi
        rotation_angle_B = np.random.uniform() * 2 * np.pi
        rotation_angle_C = np.random.uniform() * 2 * np.pi

        cosval_A = np.cos(rotation_angle_A)
        sinval_A = np.sin(rotation_angle_A)
        cosval_B = np.cos(rotation_angle_B)
        sinval_B = np.sin(rotation_angle_B)
        cosval_C = np.cos(rotation_angle_C)
        sinval_C = np.sin(rotation_angle_C)
        rotation_matrix = np.array([[cosval_B * cosval_C, -cosval_B * sinval_C, sinval_B],
                                    [sinval_A * sinval_B * cosval_C + cosval_A * sinval_C,
                                     -sinval_A * sinval_B * sinval_C + cosval_A * cosval_C, -sinval_A * cosval_B],
                                    [-cosval_A * sinval_B * cosval_C + sinval_A * sinval_C,
                                     cosval_A * sinval_B * sinval_C + sinval_A * cosval_C, cosval_A * cosval_B]])
        shape_pc = batch_data[k, ...]
        norm_pc = norm_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_norm[k, ...] = np.dot(norm_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotated_norm#, rotation_matrix

def rotate_point_cloud_norm(batch_data, norm_batch):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
          BxNx3 array, rotated normals
        Return:
          BxNx3 array, rotated batch of point clouds
          BxNx3 array, rotated normals
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_normal = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])

        shape_pc = batch_data[k, ...]
        shape_nm = norm_batch[k, ...]
        rotated_normal[k, ...] = np.dot(shape_nm.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotated_normal


def rotate_point_cloud_so3_(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape along arbitrary direction
        Input:
          BxNx3 array, original batch of point clouds
          BxNx3 array, original batch normals
        Return:
          BxNx3 array, rotated batch of point clouds
          BxNx3 array, rotated batch of normals
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_A = np.random.uniform() * 2 * np.pi
        rotation_angle_B = np.random.uniform() * 2 * np.pi
        rotation_angle_C = np.random.uniform() * 2 * np.pi

        cosval_A = np.cos(rotation_angle_A)
        sinval_A = np.sin(rotation_angle_A)
        cosval_B = np.cos(rotation_angle_B)
        sinval_B = np.sin(rotation_angle_B)
        cosval_C = np.cos(rotation_angle_C)
        sinval_C = np.sin(rotation_angle_C)
        rotation_matrix = np.array([[cosval_B * cosval_C, -cosval_B * sinval_C, sinval_B],
                                    [sinval_A * sinval_B * cosval_C + cosval_A * sinval_C,
                                     -sinval_A * sinval_B * sinval_C + cosval_A * cosval_C, -sinval_A * cosval_B],
                                    [-cosval_A * sinval_B * cosval_C + sinval_A * sinval_C,
                                     cosval_A * sinval_B * sinval_C + sinval_A * cosval_C, cosval_A * cosval_B]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data#, rotation_matrix

def rotate_point_cloud_(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
          BxNx3 array, rotated normals
        Return:
          BxNx3 array, rotated batch of point clouds
          BxNx3 array, rotated normals
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])

        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data