import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_completion_20240421_2000 import PVCNN2_VAE
import torch.distributed as dist
from utils.visualize import visualize_pointcloud_hq
import open3d as o3d
'''
some utils
'''
def save_ply(filename, point_cloud):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(filename, o3d_pcd)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    '''
    print('pNorm')
    for p in net.parameters():
        print(p.device)
    '''
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()if p.grad is not None))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points, num_views, MDM_views, MDM_multiplier, test_type='completion'):
        #_修改位置
        self.num_views = num_views
        self.MDM_views = MDM_views
        #MDM倍率修改在这
        self.MDM_multiplier = MDM_multiplier#2
        self.test_type = test_type

        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        # x_start = sv_x_list[0][:,:,self.sv_points:]
        # t = t_
        #_调试代码
        '''
        print('q_mean_variance___')
        print(x_start.shape)
        print(t.shape)
        '''

        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        # 训练参数计算过程使用
        # 两处使用
        # x_start = data_start[:,:,self.sv_points:](对列表中的每一项)
        # x_start=x_recon(单个主视角)
        # x_t = data_t(单个主视角叠加了噪声)
        # 推断过程涉及

        # _修改位置，先写成硬性3个视角
        # 当前内部无修改
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):
        # denoise_fn = self._denoise
        # data=data_t(单个主视角叠加了噪声)
        # 推断过程涉及
        # data = img_t(拼接的重复3个的单个主视角部分点云，再拼接噪声)

        # _4-2修改位置
        # _核心代码，将数据读入模型的位置
        # print(partial_cloud_slice.shape) [5, 3, 600]
        # print(noised_data_slice.shape) [5, 3, 1848]
        model_output = denoise_fn(data, t, self.sv_points)[:, :, int((self.num_views + self.MDM_multiplier) * self.sv_points):]
        # [80, 3, 1848]
        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (
                self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(model_output)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            # _修改位置
            x_recon = self._predict_xstart_from_eps(data[:, :,(self.num_views + self.MDM_views) * self.sv_points:], t=t, eps=model_output)
            # _必须修改位置，此处对于eps的计算问题很大
            # _修改位置，此处先硬性写成使用主视角
            # _修改位置
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon,x_t=data[:, :, (self.num_views + self.MDM_views) * self.sv_points:], t=t)
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        # 训练参数计算过程涉及
        # denoise_fn = self._denoise
        # data = img_t(拼接的重复3个的单个主视角部分点云，再拼接一个噪声)
        # 推断过程涉及
        # data = img_t(拼接的重复3个的单个主视角部分点云，再拼接一个噪声)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        #_修改位置
        sample = torch.cat([data[:, :, : (self.num_views + self.MDM_views) * self.sv_points], sample], dim=-1)
        #调试代码
        '''
        print('sample___')
        print(sample[:,:,:].shape)#[80, 3, 2448]
        '''

        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, partial_x, denoise_fn, shape, save_dir, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """
        # 训练的参数计算环节涉及
        # partial_x = sv_x_list[0][:,:,:opt.svpoints]  ___  选取主视角数据
        # denoise_fn = self._denoise
        # 推断过程涉及（）
        # partial_x = x[:, :, :opt.svpoints].cuda()  ___  对应单视角数据
        # 修改为partial_x = x[:, :, :opt.svpoints]
        assert isinstance(shape, (tuple, list))
        # _修改位置，修改为主视角重复三次和噪声拼接
        img_t = torch.cat([partial_x] * self.num_views, dim=-1)
        MDM_noise_shape = torch.cat([partial_x] * self.MDM_views, dim=-1).shape
        #_临时修改
        init_c = noise_fn(size=MDM_noise_shape, dtype=torch.float, device=device)
        img_t = torch.cat([img_t] + [init_c] + [noise_fn(size=shape, dtype=torch.float, device=device)], dim=-1)
        #img_t = torch.cat([img_t] + [img_t] + [noise_fn(size=shape, dtype=torch.float, device=device)], dim=-1)
        # img_t.shape [80, 3, 2448]
        # partial_x.shape [80, 3, 200]
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)
            if t % 100 == 0:
                recon_t = img_t.detach().cpu()
                recon_t = recon_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:]
                #recon_t = torch.cat([recon_t[:, :, :self.sv_points], recon_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:]],dim=-1)
                print('start recon_t randering')
                view_recon_t = recon_t.transpose(1, 2).contiguous()
                visualize_pointcloud_hq(view_recon_t[0].cpu().numpy(),out_file=f'{save_dir}/diffusion_MDM_{self.MDM_views}_recon_{t}.png')
                save_ply(f'{save_dir}/diffusion_MDM_{self.MDM_views}_recon_{t}.ply', view_recon_t[0].cpu().numpy())

            if t < 50 and t % 5 == 0 and t !=0:
                recon_t = img_t.detach().cpu()
                recon_t = recon_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:]
                #recon_t = torch.cat([recon_t[:, :, :self.sv_points],recon_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:]], dim=-1)
                print('start recon_t randering')
                view_recon_t = recon_t.transpose(1, 2).contiguous()
                visualize_pointcloud_hq(view_recon_t[0].cpu().numpy(),out_file=f'{save_dir}/diffusion_MDM_{self.MDM_views}_recon_{t}.png')
                save_ply(f'{save_dir}/diffusion_MDM_{self.MDM_views}_recon_{t}.ply', view_recon_t[0].cpu().numpy())
            if t < 5 and t !=0:
                recon_t = img_t.detach().cpu()
                recon_t = recon_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:]
                # recon_t = torch.cat([recon_t[:, :, :self.sv_points],recon_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:]], dim=-1)
                print('start recon_t randering')
                view_recon_t = recon_t.transpose(1, 2).contiguous()
                visualize_pointcloud_hq(view_recon_t[0].cpu().numpy(),out_file=f'{save_dir}/diffusion_MDM_{self.MDM_views}_recon_{t}.png')
                save_ply(f'{save_dir}/diffusion_MDM_{self.MDM_views}_recon_{t}.ply', view_recon_t[0].cpu().numpy())

        # _修改位置，修改为主视角重复三次和噪声拼接，后将维度降为1848
        assert img_t[:, :, (self.num_views + self.MDM_views) * self.sv_points:].shape == shape
        #img_t[:,:,:].shape [80, 3, 2448]
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """

        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool, noise_fn=torch.randn):
        # 训练参数计算过程使用
        # 用了三处
        # data_start = sv_x_list(多个视角部分点云)
        # data_t = data_t(主视角叠加了噪声)
        # t=t_b

        #调试代码

        #print('data_start[0].shape',data_start[0].shape)#[16, 3, 16384]
        #print('data_t.shape',data_t.shape)#[16, 3, 14336]

        #_修改位置，此处先硬性写成使用主视角
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start[0][:,:,self.sv_points:], x_t=data_t, t=t)

        B, D, N = data_start[0][:,:,:self.sv_points].shape
        data_t_in = [data_slices[:, :, :self.sv_points] for data_slices in data_start]
        #这里原来有问题，可以替换为
        init_in = [noise_fn(size=[B, D, N * self.MDM_views], dtype=torch.float, device='cuda')]
        # print(data_t_in[0].shape) [18, 3, 200]
        # print(init_in[0].shape) [18, 3, 600]
        data_t_in = torch.cat(data_t_in + init_in + [data_t], dim=-1)

        #data_t.shape [20, 3, 2048]
        #ata_t_in.shape [20, 3, 2448]
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t_in, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        #print('true_mean.shape',true_mean.shape)
        #print('true_log_variance_clipped.shape',true_log_variance_clipped.shape)
        #print('model_mean.shape',model_mean.shape)
        #print('model_log_variance.shape', model_log_variance.shape)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(model_mean.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, MDM_data, t, noise=None):
        """
        Training loss calculation
        """
        # 训练过程涉及
        # denoise_fn = self._denoise
        # data_start = sv_x_list(多个视角部分点云)
        # 推断过程涉及
        #此处可解释为将第一个视角视为主视角
        B, D, N = data_start[0].shape
        assert t.shape == torch.Size([B])
        if noise is None:
            noise = torch.randn(
                data_start[0][:, :, self.sv_points:].shape, dtype=data_start[0].dtype,device=data_start[0].device)
            #针对每个时间步的点云添加噪声并采样
        data_t = self.q_sample(x_start=data_start[0][:, :, self.sv_points:], t=t, noise=noise)

        #__关键位置——此处实现了通过部分形状对后向过程进行约束——文章的一个核心点
        # _4-2修改位置
        #_需要修改，先按强制三个视角规定
        #训练中num_views和MDM_views保持一致
        data_in = [data_slices[:, :, :self.sv_points] for data_slices in data_start]
        MDM_data_in = [MDM_data_slices[:, :, :self.sv_points] for MDM_data_slices in MDM_data]
        data_in = torch.cat(data_in + MDM_data_in + [data_t], dim=-1)
        # 调试代码
        # print(data_t.shape)# [18, 3, 14336]
        # print(data_in.shape)# [18, 3, 26624]

        #后面这些还没改
        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            # _必须要修改，解决正向过程添加噪声维度和后向过程数据维度不一致的问题（已解决，但强制使用3个视角，使用了不明确的解决方案）

            # _4-2修改位置
            # _核心代码，将数据读入模型的位置
            # eps_recon = denoise_fn(data_in, t, self.sv_points, self.num_views)[:, :, self.num_views * self.sv_points:]
            eps_recon = denoise_fn(data_in, t, self.sv_points)[:, :, int((self.num_views + self.MDM_multiplier) * self.sv_points):]
            # 调试代码
            # print(eps_recon.shape) # [18, 3, 14336] 这个不对
            # print(noise.shape) # [18, 3, 14336]
            losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start[0].shape))))

            #_可能修改位置
            #data_t是带噪声数据，如果要改损失函数应该在这

        elif self.loss_type == 'kl':
            #_修改位置，只输入主视角
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):
        # x_start = sv_x_list[0][:,:,self.sv_points:]
        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            #_需要注意，下面这行使用x_start没懂什么意思
            # _调试代码
            '''
            print('t_.shape___')
            print(t_.shape)
            '''

            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):
        # denoise_fn = self._denoise
        # x_start = sv_x_list

        #_后续修改，先按第一视角做主视角
        with torch.no_grad():
            B, T = x_start[0].shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start[0].device), torch.zeros([B, T], device=x_start[0].device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start[0].device).fill_(t)
                # Calculate VLB term at the current timestep
                # 这里原来有问题，可以替换为
                '''
                data_t = torch.cat([x_start[0][:, :, :self.sv_points], 
                                    self.q_sample(x_start=x_start[0][:, :, self.sv_points:], t=t_b)], dim=-1)
                '''
                data_t = self.q_sample(x_start=x_start[0][:, :, self.sv_points:], t=t_b)
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=data_t, t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start[0][:, :, self.sv_points:].shape
                new_mse_b = ((pred_xstart - x_start[0][:, :, self.sv_points:]) ** 2).mean(dim=list(range(1, len(pred_xstart.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start[0][:,:,self.sv_points:])
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class PVCNN2(PVCNN2_VAE):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, sv_points, embed_dim, num_views, MDM_views, MDM_multiplier, latent_multiplier, use_att ,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim,
            num_views= num_views, MDM_views = MDM_views, MDM_multiplier = MDM_multiplier,latent_multiplier = latent_multiplier, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


class Modeltrainer_c(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Modeltrainer_c, self).__init__()
        # ——————改动位置
        self.num_views = args.num_views
        self.MDM_views = args.MDM_views

        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type, args.svpoints, args.num_views, args.MDM_views, args.MDM_multiplier, args.test_type)

        self.model = PVCNN2(num_classes=args.nc, sv_points=args.svpoints, embed_dim=args.embed_dim,
                            num_views = args.num_views, MDM_views = args.MDM_views, MDM_multiplier= args.MDM_multiplier, latent_multiplier = args.latent_multiplier,
                            use_att=args.attention,
                            dropout=args.dropout, extra_feature_channels=0)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        #x0 = sv_x_list
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    # _4-2修改位置
    def _denoise(self, data, t, sv_points):
        # _4-2修改位置
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64
        out = self.model(data, t, sv_points, self.num_views, self.MDM_views)
        # _调试代码
        '''
        print('out___')
        print(out)
        print(out.shape)#2448
        '''
        return out

    def get_loss_iter(self, data, MDM_data, noises=None):
        #训练过程涉及
        #data = sv_x_list(多个视角部分点云)
        #_后续需修改位置，先按第0个为主视角改
        B, D, N = data[0].shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data[0].device)

        #print('t.shape',t.shape)
        #print('t',t)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, MDM_data=MDM_data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, partial_x, shape, save_dir, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        # 训练的参数计算环节涉及
        # partial_x = sv_x_list[0][:,:,:opt.svpoints]  ___  选取主视角数据
        # shape = sv_x_list[0][:,:,opt.svpoints:].shape
        # 推断过程涉及
        # partial_x = x[:, :, :opt.svpoints].cuda()  ___  对应单视角数据
        # shape = x[:, :, opt.svpoints:].shape
        return self.diffusion.p_sample_loop(partial_x=partial_x, denoise_fn=self._denoise, shape=shape, save_dir = save_dir, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)


    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)