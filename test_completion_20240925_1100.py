import torch
from pprint import pprint
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.utils.data

import argparse

from torch.distributions import Normal
from utils.file_utils import *
from model.completion_Trainer_20240421_2000 import Modeltrainer_c, getGradNorm

from datasets.shapenet import ShapeNetH5
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.shapenet_data_sv import *

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas



#############################################################################

def get_dataset(opt):
    tr_dataset = ShapeNetH5(dataroot=opt.dataroot_pc, split='train', npoints=opt.npoints, svpoints=opt.svpoints, category=opt.category,
                            num_views=opt.num_views, MDM_views=opt.MDM_views)
    te_dataset = ShapeNetH5(dataroot=opt.dataroot_pc, split='test', npoints=opt.npoints, svpoints=opt.svpoints, category=opt.category,
                            num_views=opt.num_views, MDM_views=opt.MDM_views)
    return tr_dataset, te_dataset


def evaluate_recon_mvr(opt, model, save_dir, logger):
    # __改动位置
    num_views = opt.num_views
    MDM_views = opt.MDM_views
    #_, test_dataset = get_mvr_dataset(opt.dataroot_pc, opt.dataroot_sv, opt.npoints, opt.category)
    _ , test_dataset = get_dataset(opt)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)
    ref = []
    samples = []
    masked = []
    k = 0

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):

        x_all = data['partial_point']
        gt_all = data['complete_point']
        #print(x_all.shape) #[4, 1, 2048, 3]
        #print(gt_all.shape) #[4, 16384, 3]

        B, V, N, C = x_all.shape

        gt = gt_all.transpose(1, 2).contiguous()
        # print(gt.shape) [6, 3, 16384]
        gt_all = gt_all[:, None, :, :].expand(-1, V, -1, -1)
        x = x_all.reshape(B * V, N, C).transpose(1, 2).contiguous()
        # print(x.shape) [6, 3, 2048]
        # print(gt_all.shape) [6, 1, 16384, 3]


        #print('m,s')
        #print(m.shape,s.shape)

        recon = model.gen_samples(x[:, :, :opt.svpoints].cuda(), gt[:, :, opt.svpoints:].shape, 'cuda',
                                      clip_denoised=False).detach().cpu()
        #print(recon.shape) #[4, 3, 26624]
        if opt.partial_set:
            recon = torch.cat([recon[:, :, :opt.svpoints], recon[:, :, (num_views + MDM_views) * opt.svpoints:]], dim=-1)
        else:
            recon = recon[:, :, (num_views + MDM_views) * opt.svpoints:]
            gt_all = gt_all[:,:,opt.svpoints:,:]
        #print(recon.shape) #[4, 3, 16384]

        recon = recon.transpose(1, 2).contiguous()
        x = x.transpose(1, 2).contiguous()
        # print(x.shape) [6, 2048, 3]


        # print(recon[:,:,:].shape)#[80, 2448, 3]
        recon_adj = recon[:, None, :, :].expand(-1, V, -1, -1)
        #x_adj = x.reshape(B,V,N,C)

        #print(gt_all.shape) #[4, 1, 16384, 3]
        #print(recon_adj.shape) #[4, 1, 16384, 3]
        #print(x_adj.shape) #[10, 1, 2048, 3]

        ref.append( gt_all)
        samples.append(recon_adj)
        #ref.append(gt_all * s + m)
        #samples.append(recon_adj * s + m)
        # masked.append(x_adj[:,:,:test_dataloader.dataset.sv_samples,:])

    ref_pcs = torch.cat(ref, dim=0)
    sample_pcs = torch.cat(samples, dim=0)
    # masked = torch.cat(masked, dim=0)

    #print(ref_pcs.shape) #[4, 1, 16384, 3]
    #print(sample_pcs.shape) #[4, 1, 16384, 3]

    B, V, N, C = ref_pcs.shape

    torch.save(ref_pcs.reshape(B, V, N, C), os.path.join(save_dir, 'recon_gt.pth'))

    # torch.save(masked.reshape(B,V, *masked.shape[2:]), os.path.join(save_dir, 'recon_masked.pth'))
    # Compute metrics
    results = EMD_CD(sample_pcs.reshape(B*V, N, C), ref_pcs.reshape(B*V, N, C), opt.metrics_bs, reduced=False)

    results = {ky: val.reshape(B,V) if val.shape == torch.Size([B*V,]) else val for ky, val in results.items()}

    pprint({key: val.mean().item() for key, val in results.items()})
    logger.info({key: val.mean().item() for key, val in results.items()})

    results['pc'] = sample_pcs
    torch.save(results, os.path.join(save_dir, 'ours_results.pth'))

    del ref_pcs, masked, results

def evaluate_saved(opt, saved_dir):
    # ours_base = '/viscam/u/alexzhou907/research/diffusion/shape_completion/output/test_chair/2020-11-04-02-10-38/syn'

    gt_pth = saved_dir + '/recon_gt.pth'
    ours_pth = saved_dir + '/ours_results.pth'
    gt = torch.load(gt_pth).permute(1,0,2,3)
    ours = torch.load(ours_pth)['pc'].permute(1,0,2,3)

    all_res = {}
    for i, (gt_, ours_) in enumerate(zip(gt, ours)):
        results = compute_all_metrics(gt_, ours_, opt.metrics_bs)

        for key, val in results.items():
            if i == 0:
                all_res[key] = val
            else:
                all_res[key] += val
        pprint(results)
    for key, val in all_res.items():
        all_res[key] = val / gt.shape[0]

    pprint({key: val.mean().item() for key, val in all_res.items()})


def main(opt):
    if opt.eval_path == '':
        exp_id = os.path.splitext(os.path.basename(__file__))[0]
        dir_id = os.path.dirname(__file__)
        output_dir = get_output_dir(dir_id, exp_id)
        print('output_dir___notset:\n',output_dir)
    else:
        output_dir = opt.eval_path
        print('output_dir___set:\n', output_dir)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Modeltrainer_c(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.cuda:
        model.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param['model_state'], strict=False)
        # model.load_state_dict(resumed_param['model_state'])
        if opt.eval_recon_mvr:
            # Evaluate generation
            evaluate_recon_mvr(opt, model, outf_syn, logger)

        if opt.eval_saved:
            evaluate_saved(opt, outf_syn)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot_pc', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--dataroot_sv', default='GenReData/')
    parser.add_argument('--category', type=str, default='chair')

    parser.add_argument('--num_views', type=int, default=3)
    # MDM control, if you want to disable the MDM, set the MDM_views MDM_multiplier latent_multiplier to 0
    parser.add_argument('--MDM_views', type=int, default=3)
    parser.add_argument('--MDM_multiplier', type=float, default=0.25)
    parser.add_argument('--latent_multiplier', type=int, default=20)
    parser.add_argument('--metrics_bs', type=int, default=8, help='metrics size')
    parser.add_argument('--partial_set', type=bool, default=1, help='use current partial point cloud')

    parser.add_argument('--test_type', default='completion')

    parser.add_argument('--bs', type=int, default=1, help='input batch size')
    #parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=3000, help='number of epochs to train for')
    #parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--eval_recon_mvr', default=True)
    parser.add_argument('--eval_saved', default=True)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--svpoints', type=int, default=200)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    #parser.add_argument('--time_num', default=1000)
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')


    parser.add_argument('--model', default='', required=True, help="path to model (to continue training)")

    '''eval'''

    parser.add_argument('--eval_path', default='')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt
if __name__ == '__main__':
    opt = parse_args()
    if opt.test_type != 'completion':
        set_seed(opt)
    main(opt)
