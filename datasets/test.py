import random
from pathlib import Path
import hashlib
import numpy as np
import os
from tqdm import tqdm


class ShapeNet_Multiview_Points(Dataset):
    def __init__(self, root_pc: str, root_views: str, cache: str, categories: list = ['chair'], split: str = 'val',
                 npoints=2048, sv_samples=800, all_points_mean=None, all_points_std=None, get_image=False, num_views=3):
        self.root = Path(root_views)
        self.split = split
        self.get_image = get_image
        params = {
            'cat': categories,
            'npoints': npoints,
            'sv_samples': sv_samples,
        }
        params = tuple(sorted(pair for pair in params.items()))
        self.cache_dir = Path(cache) / 'svpoints/{}/{}'.format('_'.join(categories),
                                                               hashlib.md5(bytes(repr(params), 'utf-8')).hexdigest())

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]
        self.npoints = npoints
        self.sv_samples = sv_samples

        self.all_points = []
        self.all_points_sv = []
        self.num_views = num_views  # 新增的参数，表示选择的视图数量

        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError('Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target)))

            sub_path_pc = os.path.join(root_pc, syn, split)
            if not os.path.isdir(sub_path_pc):
                print("目录缺失: %s" % sub_path_pc)
                continue

            self.all_mids = []
            self.imgs = []

            for x in os.listdir(sub_path_pc):
                if not x.endswith('.npy'):
                    continue
                self.all_mids.append(os.path.join(split, x[:-len('.npy')]))

            for mid in tqdm(self.all_mids):
                obj_fname = os.path.join(root_pc, syn, mid + ".npy")
                cams_pths = list((self.root / syn / mid.split('/')[-1]).glob('*_cam_params.npz'))
                if len(cams_pths) < num_views:  # 修改点：检查视图数量是否大于等于需要选择的视图数量
                    continue
                random.shuffle(cams_pths)  # 修改点：随机打乱视图列表
                point_cloud = np.load(obj_fname)
                sv_points_group = []
                img_path_group = []
                (self.cache_dir / (mid.split('/')[-1])).mkdir(parents=True, exist_ok=True)
                success = True

                for i, cp in enumerate(cams_pths[:num_views]):  # 修改点：仅选择指定数量的视图
                    p = str(cp)
                    vp = cp.split('cam_params')[0] + 'depth.png'
                    depth_minmax_pth = cp.split('_cam_params')[0] + '.npy'
                    cache_pth = str(self.cache_dir / mid.split('/')[-1] / os.path.basename(depth_minmax_pth))

                    cam_params = np.load(cp)
                    extr = cam_params['extr']
                    intr = cam_params['intr']

                    self.transform = DepthToSingleViewPoints(cam_ext=extr, cam_int=intr)

                    try:
                        sv_point_cloud = self._render(cache_pth, vp, depth_minmax_pth)
                        img_path_group.append(vp)
                        sv_points_group.append(sv_point_cloud)
                    except Exception as e:
                        print(e)
                        success = False
                        break
                if not success:
                    continue
                self.all_points_sv.append(np.stack(sv_points_group, axis=0))
                self.all_points.append(point_cloud)
                self.imgs.append(img_path_group)

        self.all_points = np.stack(self.all_points, axis=0)

        self.all_points_sv = np.stack(self.all_points_sv, axis=0)
        if all_points_mean is not None and all_points_std is not None:
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        else:
            self.all_points_mean = self.all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]
        self.all_points_sv = (self.all_points_sv - self.all_points_mean) / self.all_points_std