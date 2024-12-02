import functools

import torch.nn as nn
import torch
import numpy as np
from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention, Swish


def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8,out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    # 定义缩放系数
    r = width_multiplier

    # 判断维度，选择合适的模块
    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP

    # 判断输出通道数的类型
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]

    # 判断是否没有指定输出通道数
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        # 返回空的序列和输入通道数
        return nn.Sequential(), in_channels, in_channels

    # 存储MLP的层
    layers = []
    # 遍历输出通道数列表
    for oc in out_channels[:-1]:
        # 判断是否为dropout
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            # 计算输出通道数
            oc = int(r * oc)
            # 添加模块到layers，并更新输入通道数
            layers.append(block(in_channels, oc))
            in_channels = oc

    # 判断维度，选择合适的最后一层模块
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))

    # 返回layers和输出通道数
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    c = 0
    for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(r * out_channels)
        for p in range(num_blocks):
            attention = k % 2 == 0 and k > 0 and p == 0
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                          with_se=with_se, normalize=normalize, eps=eps)

            if c == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels+embed_dim, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
            c += 1
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    # 定义缩放系数
    r, vr = width_multiplier, voxel_resolution_multiplier
    # 计算输入通道数
    in_channels = extra_feature_channels + 3

    # 存储 sa_layers 和 sa_in_channels
    sa_layers, sa_in_channels = [], []
    # 计数器
    c = 0
    # 遍历 sa_blocks
    for conv_configs, sa_configs in sa_blocks:
        # 计数器
        k = 0
        # 存储当前 sa_block 中的模块
        sa_blocks = []
        # 添加输入通道数到 sa_in_channels
        sa_in_channels.append(in_channels)

        # 添加 SharedMLP/PVConv 到 sa_blocks
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            # 计算输出通道数
            out_channels = int(r * out_channels)
            # 循环添加模块到 sa_blocks
            for p in range(num_blocks):
                # 判断是否使用注意力机制
                attention = (c+1) % 2 == 0 and c > 0 and use_att and p == 0
                if voxel_resolution is None:
                    # 如果体素分辨率为 None，使用 SharedMLP 作为卷积块
                    block = SharedMLP
                else:
                    # 如果体素分辨率不为 None，使用 PVConv 作为卷积块
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                              attention=attention,
                                              dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                # 根据不同条件添加模块到 sa_blocks
                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k ==0:
                    sa_blocks.append(block(in_channels+embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            # 更新 extra_feature_channels
            extra_feature_channels = in_channels

        # 添加 PointNetAModule/PointNetSAModule 到 sa_blocks
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            # 如果 num_centers 为 None，使用 PointNetAModule
            block = PointNetAModule
        else:
            # 如果 num_centers 不为 None，使用 PointNetSAModule
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=extra_feature_channels+(embed_dim if k==0 else 0 ), out_channels=out_channels,
                               include_coordinates=True))
        c += 1
        # 更新 in_channels 和 extra_feature_channels
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels

        # 将 sa_blocks 添加到 sa_layers
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    # 返回 sa_layers, sa_in_channels, in_channels, num_centers
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, sv_points,
                                embed_dim=64, use_att=False, dropout=0.1,
                                with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    # 定义缩放系数
    r, vr = width_multiplier, voxel_resolution_multiplier

    # 存储 fp_layers
    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        # 存储当前 fp_block 中的模块
        fp_blocks = []
        # 计算输出通道数
        out_channels = tuple(int(r * oc) for oc in fp_configs)

        # 添加 PointNetFPModule 到 fp_blocks
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        # 添加 SharedMLP/PVConv 到 fp_blocks
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            # 计算输出通道数
            out_channels = int(r * out_channels)
            # 循环添加模块到 fp_blocks
            for p in range(num_blocks):
                # 判断是否使用注意力机制
                attention = c % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    # 如果体素分辨率为 None，使用 SharedMLP 作为卷积块
                    block = SharedMLP
                else:
                    # 如果体素分辨率不为 None，使用 PVConv 作为卷积块
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                              attention=attention, dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                # 添加模块到 fp_blocks
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        # 将 fp_blocks 添加到 fp_layers
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels


class PVCNN2Base(nn.Module):

    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.sv_points = sv_points
        self.in_channels = extra_feature_channels + 3

        #修改位置
        self.num_views = 3#临时修改

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)#创建大的层

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)#创建大的层

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,sv_points=sv_points,
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)#创建大的层

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)#创建mlp的层

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        '''
        #__改动位置
        #多视融合模块相关
        fusion_function = nn.Sequential(
            nn.Conv1d(in_channels=512*(self.num_views - 1), out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True))
        self.fusion_function = fusion_function
        '''

    def get_timestep_embedding(self, timesteps, device):
        # 确保 timesteps 的维度是 1
        assert len(timesteps.shape) == 1

        # 计算嵌入层的一半维度
        half_dim = self.embed_dim // 2

        # 计算嵌入的指数
        emb = np.log(10000) / (half_dim - 1)

        # 计算嵌入的值
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]

        # 计算嵌入矩阵
        emb = timesteps[:, None] * emb[None, :]

        # 拼接正弦和余弦部分
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # 如果嵌入维度为奇数，进行零填充
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
        # 确保嵌入形状正确
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])

        return emb

    def forward(self, inputs, t, num_views):
        #20231026sa fp模块完了都融合一次
        # 运算需要 forward 函数
        # 获取时间步的嵌入表示，并利用嵌入表示生成特征嵌入
        # temb的形状为 [B, embed_dim, N]，其中 B 表示 batch size，embed_dim 表示嵌入维度，N 表示点的数量
        features_branch, coords_branch, temb_branch = [], [], []
        in_features_list_branch = []
        coords_list_branch = []
        for j in range(num_views):
            #print('step1_____')
            temb = self.embedf(self.get_timestep_embedding(t, inputs[j].device))[:, :, None].expand(-1, -1, inputs[j].shape[-1])
            # inputs: [B, in_channels + S, N]
            # 将输入的坐标和特征分开
            coords, features = inputs[j][:, :3, :].contiguous(), inputs[j]
            coords_list, in_features_list = [], []
            for i, sa_blocks in enumerate(self.sa_layers):
                # 逐层计算特征嵌入
                in_features_list.append(features)
                coords_list.append(coords)
                if i == 0:
                    features, coords, temb = sa_blocks((features, coords, temb))
                else:
                    features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))
                #print('step2_____')
            coords_list_branch.append(coords_list)
            coords_branch.append(coords)
            temb_branch.append(temb)
            # 更新第一层的特征嵌入
            in_features_list[0] = inputs[j][:, 3:, :].contiguous()
            in_features_list_branch.append(in_features_list)
            if self.global_att is not None:
                # 全局注意力层
                features = self.global_att(features)
            features_branch.append(features)
        #print('step3_____')
        '''
        多视特征融合
        '''
        '''
        #多视特征融合模块
        fuse_features_branch = []
        for j in range(num_views):
            features_without_j = [features_branch[i] for i in range(num_views) if i != j]
            combined_features = torch.cat(features_without_j, dim=1)
            fuse_features = self.fusion_function(combined_features)
            fused_features = features_branch[j] + fuse_features
            fuse_features_branch.append(fused_features)
        '''

        #_________________________________________
        PVCNN2_out = []
        for j in range(num_views):
            coords_list, in_features_list = coords_list_branch[j], in_features_list_branch[j]
            features, coords, temb = features_branch[j], coords_branch[j], temb_branch[j]#__多视融合应当改动位置features_branch
            #print('step4_____')
            for fp_idx, fp_blocks in enumerate(self.fp_layers):
                # 逐层计算特征传播
                jump_coords = coords_list[-1 - fp_idx]
                fump_feats = in_features_list[-1 - fp_idx]
                # if fp_idx == len(self.fp_layers) - 1:
                #     jump_coords = jump_coords[:,:,self.sv_points:]
                #     fump_feats = fump_feats[:,:,self.sv_points:]

                #print('step5_____')
                features, coords, temb = fp_blocks(
                    (jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb))
            PVCNN2_out.append(self.classifier(features))
            #print('step6_____')
        # 全连接层输出
        return PVCNN2_out


