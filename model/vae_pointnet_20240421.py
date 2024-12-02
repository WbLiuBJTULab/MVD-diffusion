import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from modules import STN3d, STNkd, PointNetfeat, PointNetCls, PointNetDenseCls

class Encoder(nn.Module):
    def __init__(self, k, feature_transform=False, dropout = 0.3):
        super(Encoder, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        mu = self.fc3(x)
        sigma = torch.exp(self.fc4(x))

        if x.is_cuda:
            mu = mu.cuda()
            sigma = sigma.cuda()
            sigma_ = self.N.sample(mu.shape)
            sigma_ = sigma_.cuda()
            sigma_ = sigma * sigma_
        z = mu + sigma_
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z

class Decoder_tofeatrue(nn.Module):
    # Alter the classification model to be used as an encoder
    def __init__(self, k, sv_points, feature_transform=False, channel=3, dropout = 0.3):
        super(Decoder_tofeatrue, self).__init__()
        # self.k = k
        self.stn = STN3d(channel)

        self.linear1 = nn.Linear(k, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 2048)
        self.linear5 = nn.Linear(2048, 4096)
        self.linear6 = nn.Linear(4096, 64 * sv_points)
        # self.linear6 = nn.Linear(4096, 7500)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2048)
        self.bn5 = nn.BatchNorm1d(4096)
        self.drop1 = nn.Dropout(p=dropout)


    def forward(self, x, data_points):
        # x [5, 6]
        x = F.relu(self.bn1(self.drop1(self.linear1(x))))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = F.relu(self.bn4(self.linear4(x)))
        x = F.relu(self.bn5(self.linear5(x)))
        x = F.relu(self.linear6(x))
        # x = F.relu(self.bn5(self.linear5(x)))
        # x = F.relu(self.linear6(x))
        # x = x.reshape(x.size(0), 3, 2500)
        z_featrue = x.reshape(-1, 64, data_points)
        '''
        trans = self.stn(x)
        x = torch.bmm(trans, x)
        x = torch.sigmoid(x)
        '''
        return z_featrue

class VAE_tofeatrue(nn.Module):
    # latent_dims=num_classes 潜变量初始化需要的值（我这里应该是和视角数量相关的值）
    def __init__(self, latent_dims, sv_points):
        super(VAE_tofeatrue, self).__init__()
        self.encoder = Encoder(k=latent_dims)
        self.decoder = Decoder_tofeatrue(k=latent_dims, sv_points = sv_points)

    def forward(self, x, data_points):
        # print(x.size()) [5, 3, 600]
        z = self.encoder(x)
        # print(z.size()) [5, 6]
        z_featrue = self.decoder(z, data_points)
        # print(z.size())
        return z_featrue


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

'''
class Decoder(nn.Module):
    # Alter the classification model to be used as an encoder
    def __init__(self, k, feature_transform=False, channel=3):
        super(Decoder, self).__init__()
        # self.k = k
        self.stn = STN3d(channel)

        self.linear1 = nn.Linear(16, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 2048)
        self.linear5 = nn.Linear(2048, 4096)
        self.linear6 = nn.Linear(4096, 7500)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2048)
        self.bn5 = nn.BatchNorm1d(4096)
        self.drop1 = nn.Dropout(p=0.3)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(5, 4, 1)
        self.conv3 = torch.nn.Conv1d(4, 3, 1)
        self.bn8 = nn.BatchNorm1d(64)
        self.bn9 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = F.relu(self.bn1(self.drop1(self.linear1(x))))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = F.relu(self.bn4(self.linear4(x)))
        x = F.relu(self.bn5(self.linear5(x)))
        x = F.relu(self.linear6(x))
        # print(x.size())
        x = x.reshape(x.size(0), 3, 2500)
        # x = F.relu(self.bn8(self.conv1(x)))
        # x = F.relu(self.conv3(x))
        trans = self.stn(x)
        # print(trans.size())
        # print(x.size())
        x = torch.bmm(trans, x)
        x = torch.sigmoid(x)
        # print(x.shape)
        return x
'''

'''
class VariationalAutoencoder(nn.Module):
    # latent_dims=num_classes 潜变量初始化需要的值（我这里应该是和视角数量相关的值）
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        # print(x.size())
        z = self.encoder(x)
        z = self.decoder(z)
        # print(z.size())
        return z
'''