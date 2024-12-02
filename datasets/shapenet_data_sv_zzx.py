import warnings
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import os
import numpy as np

import hashlib
import torch
import matplotlib.pyplot as plt

import random

synset_to_label = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}

def _convert_categories(categories):
    assert categories is not None, 'List of categories cannot be empty!'
    if not (c in synset_to_label.keys() + label_to_synset.keys()
            for c in categories):
        warnings.warn('Some or all of the categories requested are not part of \
            ShapeNetCore. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets


class ShapeNet_Multiview_Points(Dataset):
    def __init__(self, root_pc:str, root_views: str, cache: str, categories: list = ['chair'], split: str= 'val',
                 npoints=2048, sv_samples=800, all_points_mean=None, all_points_std=None, get_image=False, num_views=3):
        self.root = Path(root_views)  # 视图根目录
        self.split = split  # 数据集划分
        self.get_image = get_image  # 是否获取图像
        params = {
            'cat': categories,  # 类别
            'npoints': npoints,  # 点数
            'sv_samples': sv_samples,  # 采样数
        }
        params = tuple(sorted(pair for pair in params.items()))  # 将参数排序为元组
        self.cache_dir = Path(cache) / 'svpoints/{}/{}'.format('_'.join(categories), hashlib.md5(bytes(repr(params), 'utf-8')).hexdigest())  # 缓存目录

        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录
        self.paths = []  # 路径列表
        self.synset_idxs = []  # 分类索引
        self.synsets = _convert_categories(categories)  # 将类别转换为同义词列表
        self.labels = [synset_to_label[s] for s in self.synsets]  # 将类别转换为标签列表
        self.npoints = npoints  # 点数
        self.sv_samples = sv_samples  # 采样数

        self.all_points = []  # 所有的点云数据
        self.all_points_sv = []  # 所有的采样点数据
        self.num_views = num_views  # 新增的参数，表示选择的视图数量

        # 循环遍历所需的类别
        for i in range(len(self.synsets)):

            syn = self.synsets[i]
            class_target = self.root / syn  # 类别目录
            if not class_target.exists():
                raise ValueError('Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target)))

            sub_path_pc = os.path.join(root_pc, syn, split)  # 子路径
            if not os.path.isdir(sub_path_pc):
                print("Directory missing : %s" % sub_path_pc)
                continue

            self.all_mids = []  # 所有模型ID
            self.imgs = []  # 图像列表

            '''
            这段代码通过遍历指定路径sub_path_pc中的文件，将文件名以.npy结尾的文件路径添加到self.all_mids列表中。
            '''
            for x in os.listdir(sub_path_pc):
                if not x.endswith('.npy'):
                    continue
                self.all_mids.append(os.path.join(split, x[:-len('.npy')]))

            for mid in tqdm(self.all_mids):
                # obj_fname = os.path.join(sub_path, x)
                #---使用os.path.join函数构建一个文件路径
                obj_fname = os.path.join(root_pc, syn, mid + ".npy")
                #---使用glob函数查找具有特定模式的文件路径，并将结果存储在cams_pths列表中。
                cams_pths = list((self.root/ syn/ mid.split('/')[-1]).glob('*_cam_params.npz'))
                if len(cams_pths) < num_views:  # 修改点：检查视图数量是否大于等于需要选择的视图数量2
                    continue
                random.shuffle(cams_pths)  # 修改点：随机打乱视图列表
                #---使用os.path.join函数构建一个文件路径
                point_cloud = np.load(obj_fname)
                #---创建一个空列表sv_points_group，用于存储点云数据。创建一个空列表img_path_group，用于存储图像路径。
                sv_points_group = []
                img_path_group = []
                #---使用mkdir函数创建一个缓存目录
                (self.cache_dir / (mid.split('/')[-1])).mkdir(parents=True, exist_ok=True)
                success = True

                for i, cp in enumerate(cams_pths[:num_views]):  # 修改点：仅选择指定数量的视图
                    #将路径字符串进行一些处理，并将处理结果存储在相应的变量中
                    p = str(cp)
                    vp = cp.split('cam_params')[0] + 'depth.png'
                    depth_minmax_pth = cp.split('_cam_params')[0] + '.npy'
                    cache_pth = str(self.cache_dir / mid.split('/')[-1] / os.path.basename(depth_minmax_pth) )
                    #cache_pth为缓存的路径字符串

                    #从路径cp加载一个.npy文件，并将其存储在变量cam_params中
                    cam_params = np.load(cp)
                    extr = cam_params['extr']
                    intr = cam_params['intr']

                    #根据从.npy文件中加载的cam_params参数，使用DepthToSingleViewPoints函数，并将其存储在self.transform变量中
                    self.transform = DepthToSingleViewPoints(cam_ext=extr, cam_int=intr)

                    try:
                        #记录相关参数
                        sv_point_cloud = self._render(cache_pth, vp, depth_minmax_pth)
                        img_path_group.append(vp)
                        sv_points_group.append(sv_point_cloud)
                    #统计异常
                    except Exception as e:
                        print(e)
                        success=False
                        break
                if not success:
                    continue
                self.all_points_sv.append(np.stack(sv_points_group, axis=0))
                self.all_points.append(point_cloud)
                self.imgs.append(img_path_group)

        self.all_points = np.stack(self.all_points, axis=0)

        self.all_points_sv = np.stack(self.all_points_sv, axis=0)
        if all_points_mean is not None and all_points_std is not None:  # 使用加载的数据集统计数据
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        else:  # 标准化整个数据集
            self.all_points_mean = self.all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:,:10000]
        self.test_points = self.all_points[:,10000:]
        self.all_points_sv = (self.all_points_sv - self.all_points_mean) / self.all_points_std

    def get_pc_stats(self, idx):

        return self.all_points_mean.reshape(1,1, -1), self.all_points_std.reshape(1,1, -1)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.all_points)

    def __getitem__(self, index):

        # 获取训练点集
        tr_out = self.train_points[index]

        # 从训练点集中随机选择 npoints 个点
        tr_idxs = np.random.choice(tr_out.shape[0], self.npoints)
        tr_out = tr_out[tr_idxs, :]

        # 获取测试的真实点云（ground truth points）
        gt_points = self.test_points[index][:self.npoints]

        # 使用 get_pc_stats 方法计算数据的均值（mean）和标准差（std）
        m, s = self.get_pc_stats(index)

        # 获取全部点云数据 all_points_sv
        sv_points = self.all_points_sv[index]

        # 选择子样本点（子采样点）构成 idxs
        idxs = np.arange(0, sv_points.shape[-2])[
               :self.sv_samples]  # np.random.choice(sv_points.shape[0], 500, replace=False)

        # 创建数据张量 data，通过将选中的 sv_points 和零填充的点进行拼接，以填满剩余空间
        data = torch.cat([torch.from_numpy(sv_points[:, idxs]).float(),
                          torch.zeros(sv_points.shape[0], self.npoints - idxs.shape[0], sv_points.shape[2])], dim=1)

        # 创建掩码张量 masks，与 data 具有相同的形状，并将对应选中点的索引设置为 1
        masks = torch.zeros_like(data)
        masks[:, :idxs.shape[0]] = 1

        # 创建包含结果数据和元数据的字典 res
        res = {'train_points': torch.from_numpy(tr_out).float(),
               'test_points': torch.from_numpy(gt_points).float(),
               'sv_points': data,
               'masks': masks,
               'std': s,
               'mean': m,
               'idx': index,
               'name': self.all_mids[index]
               }

        # 如果数据集划分不是 'train' 并且 get_image 标志为 True
        if self.split != 'train' and self.get_image:

            # 初始化一个列表来存储图像数据张量
            img_lst = []

            # 遍历 all_points_sv 加载相应的图像数据
            for n in range(self.all_points_sv.shape[1]):
                # 加载图像数据，将其转换为张量，然后重新排列维度以匹配预期的形状
                img = torch.from_numpy(plt.imread(self.imgs[index][n])).float().permute(2, 0, 1)[:3]

                # 将图像张量添加到列表中
                img_lst.append(img)

            # 将图像张量沿第一个维度堆叠，创建单个张量
            img = torch.stack(img_lst, dim=0)

            # 将图像张量添加到字典中
            res['image'] = img

        return res

    def _render(self, cache_path, depth_pth, depth_minmax_pth):
        # 如果缓存路径中对应的颜色图像文件不存在，并且缓存文件存在，则删除缓存文件
        # if not os.path.exists(cache_path.split('.npy')[0] + '_color.png') and os.path.exists(cache_path):
        #
        #     os.remove(cache_path)

        # 如果缓存文件存在，则从缓存文件中加载数据
        if os.path.exists(cache_path):
            data = np.load(cache_path)
        else:
            # 否则，通过调用 transform 将颜色图像和深度图像的路径作为参数传递，获取数据和深度信息
            data, depth = self.transform(depth_pth, depth_minmax_pth)

            # 确保数据点的数量大于 600
            assert data.shape[0] > 600, 'Only {} points found'.format(data.shape[0])

            # 随机选择 600 个数据点，使用 replace=False 表示不重复选择
            data = data[np.random.choice(data.shape[0], 600, replace=False)]

            # 将新的数据保存到缓存文件中
            np.save(cache_path, data)

        # 返回数据
        return data


class DepthToSingleViewPoints(object):
    '''
    render a view then save mask
    '''
    def __init__(self, cam_ext, cam_int):

        # 初始化函数将相机外参（cam_ext）和相机内参（cam_int）变换为矩阵形式
        self.cam_ext = cam_ext.reshape(4,4)
        self.cam_int = cam_int.reshape(3,3)


    def __call__(self, depth_pth, depth_minmax_pth):

        # 加载深度范围信息
        depth_minmax = np.load(depth_minmax_pth)

        # 加载深度图像
        depth_img = plt.imread(depth_pth)[...,0]

        # 创建掩码，将深度图像中为 0 的像素点设为 -1，其余设为 1
        mask = np.where(depth_img == 0, -1.0, 1.0)

        # 对深度图像进行翻转和归一化处理
        depth_img = 1 - depth_img
        depth_img = (depth_img * (np.max(depth_minmax) - np.min(depth_minmax)) + np.min(depth_minmax)) * mask

        # 创建 o3d.camera.PinholeCameraIntrinsic 对象 intr，用于保存相机参数
        intr = o3d.camera.PinholeCameraIntrinsic(depth_img.shape[0], depth_img.shape[1],
                                                 self.cam_int[0, 0], self.cam_int[1, 1], self.cam_int[0,2],
                                                 self.cam_int[1,2])

        # 创建 o3d.geometry.Image 对象 depth_im，用于保存深度图像
        depth_im = o3d.geometry.Image(depth_img.astype(np.float32, copy=False))

        # 创建 o3d.geometry.PointCloud 对象 pcd，从深度图像 depth_im、相机参数 intr 和相机外参 self.cam_ext 中生成点云数据
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_im, intr, self.cam_ext, depth_scale=1.)

        # 将点云数据 pcd 转换为 numpy 数组
        pc =  np.asarray(pcd.points)

        # 返回点云数据 pc 和深度图像 depth_img
        return pc, depth_img

    def __repr__(self):
        return 'MeshToMaskedVoxel_'+str(self.radius)+str(self.resolution)+str(self.elev )+str(self.azim)+str(self.img_size )

