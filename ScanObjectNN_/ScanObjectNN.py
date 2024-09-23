"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset


import open3d as o3d

def compute_normal(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    #pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #pcd.orient_normals_to_align_with_direction()
    pcd.orient_normals_towards_camera_location()
    fp = np.asarray(pcd.points)
    fn = np.asarray(pcd.normals)
    fpn = np.concatenate((fp, fn), 1)
    return fpn


def compute_norm(batch_data):
    datan = np.zeros((batch_data.shape[0], batch_data.shape[1], 6), dtype=np.float32)
    for k in range(batch_data.shape[0]):
        shape_pc = batch_data[k, ...]
        datan[k, ...] = compute_normal(shape_pc.reshape(-1, 3))

    return datan

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(args, partition):
    #download()
    BASE_DIR = args.data_root
    if args.dataset=='OBJ_ONLY':
        h5_name = BASE_DIR+'/h5_files/main_split_nobg/'+ partition +'_objectdataset.h5'
    elif args.dataset=='OBJ_BG':
        h5_name = BASE_DIR + '/h5_files/main_split/' + partition + '_objectdataset.h5'
    elif args.dataset == 'PB_T50_RS':
        h5_name = BASE_DIR + '/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    else:
        raise ValueError('Data not recognized')

    all_data = []
    all_label = []
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, args, partition='training'):
        self.data, self.label = load_scanobjectnn_data(args, partition)
        self.num_points = args.num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        pointcloud=compute_normal(pointcloud)
        if self.partition == 'training':
            #pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(2049)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
