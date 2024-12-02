import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d

class ShapeNet_origin(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """

    def __init__(self, dataroot, split, category, npoints, svpoints, input_dim=3):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane": "02691156",  # plane
            "cabinet": "02933112",  # dresser
            "car": "02958343",
            "chair": "03001627",
            "lamp": "03636649",
            "sofa": "04256520",
            "table": "04379243",
            "vessel": "04530566",  # boat

            # alis for some seen categories
            "boat": "04530566",  # vessel
            "couch": "04256520",  # sofa
            "dresser": "02933112",  # cabinet
            "airplane": "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus": "02924116",
            "bed": "02818832",
            "bookshelf": "02871439",
            "bench": "02828884",
            "guitar": "03467517",
            "motorbike": "03790512",
            "skateboard": "04225987",
            "pistol": "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.npoints = npoints
        self.svpoints = svpoints
        self.input_dim = input_dim
        self.partial_paths, self.complete_paths = self._load_data()

    def __getitem__(self, index):
        if self.split == 'train':
            view_data = []
            for i in range(8):
                partial_path = self.partial_paths[index].format(i)
                partial_pc = self.random_sample(self.read_point_cloud(partial_path), self.svpoints)
                # 归一化处理
                # partial_pc = (partial_pc - self.mean) / self.std
                partial_tensor = torch.from_numpy(partial_pc)
                view_data.append(partial_tensor)
            views_tensor = torch.stack(view_data, dim=0)

            complete_path = self.complete_paths[index]
            complete_pc = self.random_sample(self.read_point_cloud(complete_path), self.npoints)
            # 归一化处理
            # complete_pc = (complete_pc - self.mean) / self.std
            complete_tensor = torch.from_numpy(complete_pc)

        else:
            partial_path = self.partial_paths[index]
            partial_pc = self.random_sample(self.read_point_cloud(partial_path), self.svpoints)
            # 归一化处理
            # partial_pc = (partial_pc - self.mean) / self.std
            views_tensor = torch.from_numpy(partial_pc).unsqueeze(0)

            complete_path = self.complete_paths[index]
            complete_pc = self.random_sample(self.read_point_cloud(complete_path), self.npoints)
            # 归一化处理
            # complete_pc = (complete_pc - self.mean) / self.std
            complete_tensor = torch.from_numpy(complete_pc)

        res = {
            'partial_point': views_tensor,
            'complete_point': complete_tensor,
            'idx': index,
        }

        return res

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))

        partial_paths, complete_paths = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            if self.split == 'train':
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '_{}.ply'))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.ply'))
            complete_paths.append(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.ply'))

        return partial_paths, complete_paths

    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)

    def random_sample(self, pc, n):
        if pc.shape[0] == 0:
            raise ValueError("Point cloud is empty.")
        if pc.shape[0] >= n:
            idx = np.random.choice(pc.shape[0], n, replace=False)
        else:
            idx = np.arange(pc.shape[0])
            if pc.shape[0] > 0:
                idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
            else:
                raise ValueError("Not enough points to sample.")
        return pc[idx, :]