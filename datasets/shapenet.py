import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d

import h5py


class ShapeNet(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """
    
    def __init__(self, dataroot, split, category, npoints, svpoints , input_dim=3, num_views=3, MDM_views=3):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
            
            # alis for some seen categories
            "boat"      : "04530566",  # vessel
            "couch"     : "04256520",  # sofa
            "dresser"   : "02933112",  # cabinet
            "airplane"  : "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.npoints = npoints
        self.svpoints = svpoints
        self.num_views = min(max(num_views, MDM_views), 8)
        self.input_dim = input_dim
        self.partial_paths, self.complete_paths = self._load_data()
        self.mean, self.std = self._compute_mean_std()

    def __getitem__(self, index):
        if self.split == 'train':
            view_data = []
            view_indices = np.random.choice(8, self.num_views, replace=False)  # 随机选择视图索引
            for i in view_indices:
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
            #complete_pc = (complete_pc - self.mean) / self.std
            complete_tensor = torch.from_numpy(complete_pc)

        else:
            partial_path = self.partial_paths[index]
            partial_pc = self.random_sample(self.read_point_cloud(partial_path), self.svpoints)
            # 归一化处理
            #partial_pc = (partial_pc - self.mean) / self.std
            views_tensor = torch.from_numpy(partial_pc).unsqueeze(0)

            complete_path = self.complete_paths[index]
            complete_pc = self.random_sample(self.read_point_cloud(complete_path), self.npoints)
            # 归一化处理
            #complete_pc = (complete_pc - self.mean) / self.std
            complete_tensor = torch.from_numpy(complete_pc)

        res = {
              'partial_point': views_tensor,
              'complete_point': complete_tensor,
              'idx': index,
              'complete_mean': self.mean.reshape(1,1, -1),
              'complete_std': self.std.reshape(1,1, -1)
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
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]

    # 修改位置
    def _compute_mean_std(self):
        complete_points = []
        print(f"Calculating")
        for path in self.complete_paths:
            pc = self.read_point_cloud(path)
            #print('complete_paths', pc.shape) #(16384, 3)
            complete_points.append(pc)
        # 将列表中的数组合并为一个大数组
        complete_points = np.stack(complete_points, axis=0)

        # 计算均值和标准差
        complete_mean = complete_points.reshape(-1, 3).mean(axis=0).reshape(1, 3)
        complete_std = complete_points.reshape(-1).std(axis=0).reshape(1, 1)

        print(f"Calculated Complete Mean: {complete_mean}, Complete Std: {complete_std}")
        return complete_mean, complete_std

class ShapeNetH5(data.Dataset):
    def __init__(self, dataroot, split='train', category='all', npoints=16384, svpoints=2048, input_dim=3,
                 novel_input=False, novel_input_only=False, num_views=3, MDM_views=3):
        assert split in ['train', 'valid', 'test'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane": 0,
            "cabinet": 1,
            "car": 2,
            "chair": 3,
            "lamp": 4,
            "sofa": 5,
            "table": 6,
            "vessel": 7,
            # unseen categories
            "bus": 8,
            "bed": 9,
            "bookshelf": 10,
            "bench": 11,
            "guitar": 12,
            "motorbike": 13,
            "skateboard": 14,
            "pistol": 15,
        }

        self.category = category
        self.category_id = self.cat2id.get(self.category, -1)
        self.svpoints = svpoints
        self.split = split
        self.npoints = npoints
        self.input_dim = input_dim

        # 随机视角数量的控制：取MDM_views和num_views的较大值，并限制最大值为26
        self.num_views = min(max(num_views, MDM_views), 26)

        if split == 'train':
            self.input_path = os.path.join(dataroot, 'mvp_train_input.h5')
            self.gt_path = os.path.join(dataroot, f'mvp_train_gt_{npoints}pts.h5')
        else:
            self.input_path = os.path.join(dataroot, 'mvp_test_input.h5')
            self.gt_path = os.path.join(dataroot, f'mvp_test_gt_{npoints}pts.h5')

        with h5py.File(self.input_path, 'r') as input_file:
            self.input_data = np.array(input_file['incomplete_pcds'])
            self.labels = np.array(input_file['labels'])
            self.novel_input_data = np.array(input_file['novel_incomplete_pcds'])
            self.novel_labels = np.array(input_file['novel_labels'])

        with h5py.File(self.gt_path, 'r') as gt_file:
            self.gt_data = np.array(gt_file['complete_pcds'])
            self.novel_gt_data = np.array(gt_file['novel_complete_pcds'])

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        if self.category != 'all':
            self._filter_by_category(self.category_id)

        print(f"Input data shape: {self.input_data.shape}")
        print(f"GT data shape: {self.gt_data.shape}")
        print(f"Labels shape: {self.labels.shape}")

        '''
        self.len = self.input_data.shape[0] // 26 if self.split == 'train' else self.input_data.shape[0]
        '''
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        '''
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        '''

        if self.split == 'train':
            object_index = index // 26  # 确定完整点云的索引
            view_index = index % 26  # 当前的视角索引

            # 按顺序选择一个当前视角
            sequential_partial_pc = self.random_sample(self.input_data[index], self.svpoints)
            sequential_partial_tensor = torch.from_numpy(sequential_partial_pc).float()

            # 在剩余的25个视角中随机挑选 num_views - 1 个
            available_views = list(set(range(26)) - {view_index})
            random_views = np.random.choice(available_views, self.num_views - 1, replace=False)

            view_data = [sequential_partial_tensor]
            for i in random_views:
                partial_pc = self.random_sample(self.input_data[object_index * 26 + i], self.svpoints)
                partial_tensor = torch.from_numpy(partial_pc).float()
                view_data.append(partial_tensor)
            views_tensor = torch.stack(view_data, dim=0)
        else:
            object_index = index // 26  # 确定完整点云的索引
            view_index = index % 26  # 当前的视角索引
            partial_pc = self.random_sample(self.input_data[index], self.svpoints)
            views_tensor = torch.from_numpy(partial_pc).unsqueeze(0).float()

        complete_pc = self.random_sample(self.gt_data[object_index], self.npoints)
        complete_tensor = torch.from_numpy(complete_pc).float()

        label = int(self.labels[index])

        res = {
            'partial_point': views_tensor,
            'complete_point': complete_tensor,
            'label': label,
            'idx': object_index,
            'view_idx': view_index
        }
        return res


    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
        return pc[idx[:n]]

    def _filter_by_category(self, target_label):
        # 选择每个样本的第一个视角标签作为样本标签
        sample_labels = self.labels[::26]  # 每个样本有26个视角，取第一个视角的标签
        sample_mask = sample_labels == target_label  # 生成样本级别的掩码

        # 过滤 gt_data，使其与样本数量一致
        self.gt_data = self.gt_data[sample_mask]

        # 如果 input_data 存在，重复掩码以适应视角数量
        if hasattr(self, 'input_data'):
            self.input_data = self.input_data[sample_mask.repeat(26)]
            self.labels = self.labels[sample_mask.repeat(26)]

        # 处理 novel 数据
        if hasattr(self, 'novel_input_data') and self.novel_input_data.size > 0:
            novel_sample_labels = self.novel_labels[::26]
            novel_sample_mask = novel_sample_labels == target_label
            self.novel_input_data = self.novel_input_data[novel_sample_mask.repeat(26)]
            self.novel_labels = self.novel_labels[novel_sample_mask.repeat(26)]
            self.novel_gt_data = self.novel_gt_data[novel_sample_mask]

'''
    def __getitem__(self, index):
        # 修改3: 统一索引和随机视角选择
        actual_index = index % self.len  # 确定对应的样本索引
        chosen_views = np.random.choice(26, self.num_views, replace=False)
        view_data = []
        for i in chosen_views:
            partial_pc = self.random_sample(self.input_data[actual_index * 26 + i], self.svpoints)
            partial_tensor = torch.from_numpy(partial_pc).float()
            view_data.append(partial_tensor)
        views_tensor = torch.stack(view_data, dim=0)

        complete_pc = self.random_sample(self.gt_data[actual_index], self.npoints)
        complete_tensor = torch.from_numpy(complete_pc).float()

        label = int(self.labels[actual_index * 26])  # 修改4: Label与样本索引一致

        res = {
            'partial_point': views_tensor,
            'complete_point': complete_tensor,
            'label': label,
            'idx': actual_index,
            'view_idx': -1  # 训练时视角索引无意义
        }
        return res
    '''