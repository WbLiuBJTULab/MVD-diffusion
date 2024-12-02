# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:21:32
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import os
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO

label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data

code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}

def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class MyShapeNetDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, root='/data1/xp/PCN', phase='train', categories=None):
        assert phase in {'train', 'val', 'test'}
        self.phase = phase
        base_dir = os.path.join(root, phase)
        if categories is None:
            self.taxomony_ids = list(code_mapping.values())
        else:
            taxomony_ids = []
            for c in categories:
                taxomony_ids.append(code_mapping[c])
            self.taxomony_ids = taxomony_ids

        all_taxomony_ids = []
        all_model_ids = []
        all_pcds_partial = []
        all_pcds_gt = []

        for t_id in self.taxomony_ids:
            gt_dir = os.path.join(base_dir, 'complete', t_id)
            partial_dir = os.path.join(base_dir, 'partial', t_id)
            model_ids = os.listdir(partial_dir)
            all_taxomony_ids.extend([t_id for i in range(len(model_ids))])
            all_model_ids.extend(model_ids)
            all_pcds_gt.extend([os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))])
            all_pcds_partial.extend([os.path.join(partial_dir, f) for f in sorted(os.listdir(partial_dir))])

        self.taxomony_ids = all_taxomony_ids
        self.model_ids = all_model_ids
        self.path_partial = all_pcds_partial
        self.path_gt = all_pcds_gt
        self.LEN = len(self.model_ids)
        self.transform = UpSamplePoints({'n_points': 2048})

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        if self.phase == 'test':
            partial = read_ply(self.path_partial[index]).astype(np.float32)
        else:
            idx_partial = random.randint(0, 7)
            partial = read_ply(os.path.join(self.path_partial[index], '0{}.pcd'.format(idx_partial))).astype(np.float32)
        partial = self.transform(partial)
        gt = read_ply(self.path_gt[index]).astype(np.float32)
        idx_random_complete = random.randint(0, self.LEN - 1)
        random_complete = read_ply(self.path_gt[idx_random_complete]).astype(np.float32)
        data = {
            'X': torch.from_numpy(partial).float(),
            'Y': torch.from_numpy(random_complete).float(),
            'X_GT': torch.from_numpy(gt).float()
        }
        return self.taxomony_ids[index], self.model_ids[index], data






class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        """
        初始化Dataset
        :param options: 包含配置选项的字典
        :param file_list: 文件路径列表
        :param transforms: 数据转换操作（可选）
        """
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        """
        获取一个样本数据
        :param idx: 样本的索引
        :return: 样本的 taxonomy_id, model_id 和处理后的数据
        """
        sample = self.file_list[idx]
        data = {}
        taxonomy_id = sample['taxonomy_id']
        model_id = sample['model_id']

        # 获取 n_renderings 和 num_views 参数
        n_renderings = self.options.get('n_renderings', 1)
        num_views = min(self.options.get('num_views', 1), n_renderings)

        # 仅在训练集上进行随机视角选择
        if self.options.get('shuffle', False):
            rand_indices = random.sample(range(n_renderings), num_views)
        else:
            # 非训练模式下，直接选择第一个视角
            rand_indices = list(range(num_views))

        # 遍历 required_items 并加载数据
        for ri in self.options.get('required_items', []):
            file_path = sample.get(f'{ri}_path')

            # 如果是 partial_cloud 且是多个视角路径
            if ri == 'partial_cloud' and isinstance(file_path, list):
                partial_clouds = []
                for idx in rand_indices:
                    partial_cloud = IO.get(file_path[idx]).astype(np.float32)

                    # 在堆叠前对每个视角的 partial_cloud 进行处理
                    single_data = {'partial_cloud': partial_cloud}
                    if self.transforms:
                        single_data = self.transforms(single_data)

                    # 处理后的数据添加到列表
                    partial_clouds.append(single_data['partial_cloud'])

                # 堆叠所有的视角
                data[ri] = np.stack(partial_clouds) if len(partial_clouds) > 1 else partial_clouds[0]
            else:
                # 对于其他项目，例如 gtcloud，直接加载数据
                data[ri] = IO.get(file_path).astype(np.float32)

        return taxonomy_id, model_id, data, idx


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.category_file_path_pcn) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.n_renderings if subset == DatasetSubset.TRAIN else 1
        num_views = self.cfg.num_views
        MDM_views = self.cfg.MDM_views
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'num_views': num_views,
            'MDM_views': MDM_views,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.npoints
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.npoints
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':

                    gt_path = cfg.complete_points_path_pcn % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': gt_path.replace('complete', 'partial'),
                    'gtcloud_path': gt_path})
                else:
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': [
                            cfg.partial_points_path_pcn % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gtcloud_path':
                            cfg.complete_points_path_pcn % (subset, dc['taxonomy_id'], s),
                    })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

#New adding
class ShapeNetDataLoader_Single(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetDataLoader_Single, self).__init__(cfg)

        category_id = code_mapping.get(cfg.category)
        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == category_id]

class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.category_file_path_completion3d) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        num_views = self.cfg.num_views
        MDM_views = self.cfg.MDM_views
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']
        return Dataset({
            'num_views': num_views,
            'MDM_views': MDM_views,
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.svpoints
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.partial_points_path_completion3d % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.complete_points_path_completion3d % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class Completion3DPCCTDataLoader(Completion3DDataLoader):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDataLoader, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343', '04379243'} # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] in cat_set]


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'Completion3DPCCT': Completion3DPCCTDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'ShapeNetSingle': ShapeNetDataLoader_Single,
}  # yapf: disable

