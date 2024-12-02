import functools

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import numpy as np
from modules.pointnet_utils import PointNetEncoder, PointNetCls, PointNetDenseCls
from modules.pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


class Encoder(nn.Module):
    def __init__(self, k, feature_transform=False):
        super(Encoder, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
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


class Decoder(nn.Module):
    # Alter the classification model to be used as an encoder
    def __init__(self, k, feature_transform=False):
        super(Decoder, self).__init__()
        # self.k = k
        self.stn = STN3d()

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


# vae = VariationalAutoencoder(latent_dims=num_classes)
class VariationalAutoencoder(nn.Module):
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

#nn.functional.gumbel_softmax(logits, tau=temperature, hard=False)


def latent_features(self, data_loader, return_labels=False):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    # 直接调用另一个项目的代码，应该是用于可视化的部分
    self.network.eval()
    N = len(data_loader.dataset)
    features = np.zeros((N, self.gaussian_size))
    if return_labels:
        true_labels = np.zeros(N, dtype=np.int64)
    start_ind = 0
    with torch.no_grad():
        for (data, labels) in data_loader:
            if self.cuda == 1:
                data = data.cuda()
            # flatten data
            data = data.view(data.size(0), -1)
            out = self.network.inference(data, self.gumbel_temp, self.hard_gumbel)
            latent_feat = out['mean']
            end_ind = min(start_ind + data.size(0), N + 1)

            # return true labels
            if return_labels:
                true_labels[start_ind:end_ind] = labels.cpu().numpy()
            features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()
            start_ind += data.size(0)
    if return_labels:
        return features, true_labels
    return features


def reconstruct_data(self, data_loader, sample_size=-1):
    """Reconstruct Data

    Args:
        data_loader: (DataLoader) loader containing the data
        sample_size: (int) size of random data to consider from data_loader

    Returns:
        reconstructed: (array) array containing the reconstructed data
    """
    # 直接调用另一个项目的代码，应该是用于可视化的部分
    self.network.eval()

    # sample random data from loader
    indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
    test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size,
                                                     sampler=SubsetRandomSampler(indices))

    # obtain values
    it = iter(test_random_loader)
    test_batch_data, _ = it.next()
    original = test_batch_data.data.numpy()
    if self.cuda:
        test_batch_data = test_batch_data.cuda()

        # obtain reconstructed data
    out = self.network(test_batch_data, self.gumbel_temp, self.hard_gumbel)
    reconstructed = out['x_rec']
    return original, reconstructed.data.cpu().numpy()

def plot_latent_space(self, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    # 直接调用另一个项目的代码，应该是用于可视化的部分
    # obtain the latent features
    features = self.latent_features(data_loader)

    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
    plt.colorbar()
    if (save):
        fig.savefig('latent_space.png')
    return fig


def random_generation(self, num_elements=1):
    """Random generation for each category

    Args:
        num_elements: (int) number of elements to generate

    Returns:
        generated data according to num_elements
    """
    # 直接调用另一个项目的代码，应该是用于可视化的部分
    # categories for each element
    arr = np.array([])
    for i in range(self.num_classes):
        arr = np.hstack([arr, np.ones(num_elements) * i])
    indices = arr.astype(int).tolist()

    categorical = F.one_hot(torch.tensor(indices), self.num_classes).float()

    if self.cuda:
        categorical = categorical.cuda()

    # infer the gaussian distribution according to the category
    mean, var = self.network.generative.pzy(categorical)

    # gaussian random sample by using the mean and variance
    noise = torch.randn_like(var)
    std = torch.sqrt(var)
    gaussian = mean + noise * std

    # generate new samples with the given gaussian
    generated = self.network.generative.pxz(gaussian)

    return generated.cpu().detach().numpy()
