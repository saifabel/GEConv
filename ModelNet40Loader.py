import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]

def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    normals = f['normal'][:]
    return data, label, normals
    
class ModelNet40Cls(data.Dataset):

    def __init__(
            self, num_points, root, transforms=None, train=True
    ):
        super().__init__()

        self.transforms = transforms

        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)

        self.train, self.num_points = train, num_points
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list, normal_list = [], [], []
        for f in self.files:
            points, labels, normals = _load_data_file(os.path.join(root, f))
            point_list.append(points)
            label_list.append(labels)
            normal_list.append(normals)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        self.normals = np.concatenate(normal_list, 0)
    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        current_normals = self.normals[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)
        
        return current_points[:self.num_points, :], label, current_normals[:self.num_points, :]

    def __len__(self):
        return self.points.shape[0]
