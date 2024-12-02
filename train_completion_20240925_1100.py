import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.completion_Trainer_20240421_2000 import Modeltrainer_c, getGradNorm
import torch.distributed as dist
from datasets.shapenet import ShapeNetH5
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.shapenet_data_sv import ShapeNet_Multiview_Points
'''
——————————————zzx debug code
'''
def zzxdebugcode(objectname , objectinformation):
    print(objectname, '____')
    print(objectinformation)
    #print(objectinformation.type())
    print('____________')


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


def get_dataset(opt):
    #使用MVP数据集
    tr_dataset = ShapeNetH5(dataroot=opt.dataroot_pc, split='train', npoints=opt.npoints, svpoints=opt.svpoints, category=opt.category,
                            num_views=opt.num_views, MDM_views=opt.MDM_views)
    te_dataset = ShapeNetH5(dataroot=opt.dataroot_pc, split='test', npoints=opt.npoints, svpoints=opt.svpoints, category=opt.category,
                            num_views=opt.num_views, MDM_views=opt.MDM_views)

    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train(gpu, opt, output_dir, noises_init):

    set_seed(opt)
    logger = setup_logging(output_dir)


    num_views = opt.num_views
    MDM_views = opt.MDM_views

    if opt.distribution_type == 'multi':
        should_diag = gpu == 0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter =  int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)


    ''' data '''
    train_dataset, _ = get_dataset(opt)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

    '''
    create networks
    '''
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Modeltrainer_c(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)

    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    optimizer= optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0


    for epoch in range(start_epoch, opt.niter):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)

        for i, data in enumerate(dataloader):

            # print(data_train_points.shape) [10, 16384, 3]
            # print(data_sv_points.shape) [10, 2048, 3]
            # print(data_idx.shape) [10]
            if num_views != MDM_views:
                randind_max = max(num_views, MDM_views)
            else:
                randind_max = num_views
            randind_num = np.random.choice(randind_max, size=(num_views,))  # 8 views
            randind_MDM = np.random.choice(randind_max, size=(MDM_views,))  # 8 views

            x = data['complete_point'].transpose(1,2) #[10, 3, 16384]
            sv_x_list = []
            sv_MDM_list = []
            for j in range(num_views):
                # print(data['partial_point'].shape) [18, 8, 2048, 3]
                sv_x_list.append(data['partial_point'][:, randind_num[j]].transpose(1,2)) #[10, 3, 2048]
                sv_x_list[j] = torch.cat((sv_x_list[j], x[:, :, opt.svpoints:]), dim =-1)
                # print(sv_x_list[j].shape) [18, 3, 16384]
            for j in range(MDM_views):
                # print(data['partial_point'].shape) [18, 8, 2048, 3]
                sv_MDM_list.append(data['partial_point'][:, randind_MDM[j]].transpose(1,2)) #[10, 3, 2048]
                # print(sv_x_list[j].shape) [18, 3, 16384]

            noises_batch = noises_init[data['idx']].transpose(1, 2)

            '''
            train diffusion
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                for j in range(num_views):
                    sv_x_list[j] = sv_x_list[j].cuda(gpu)
                for j in range(MDM_views):
                    sv_MDM_list[j] = sv_MDM_list[j].cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
            elif opt.distribution_type == 'single':
                for j in range(num_views):
                    sv_x_list[j] = sv_x_list[j].cuda()
                for j in range(MDM_views):
                    sv_MDM_list[j] = sv_MDM_list[j].cuda()
                noises_batch = noises_batch.cuda()


            loss = model.get_loss_iter(sv_x_list, sv_MDM_list, noises_batch).mean()
            #zzxdebugcode('loss',loss)
            optimizer.zero_grad()
            loss.backward()

            netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             'netpNorm: {:>10.2f},   netgradNorm: {:>10.2f}     '
                             .format(
                        epoch, opt.niter, i, len(dataloader),loss.item(),
                    netpNorm, netgradNorm,))

        if (epoch + 1) % opt.diagIter == 0 and should_diag:

            logger.info('Diagnosis:')
            x_range = [x.min().item(), x.max().item()]
            kl_stats = model.all_kl(sv_x_list)
            logger.info('      [{:>3d}/{:>3d}]    '
                         'x_range: [{:>10.4f}, {:>10.4f}],   '
                         'total_bpd_b: {:>10.4f},    '
                         'terms_bpd: {:>10.4f},  '
                         'prior_bpd_b: {:>10.4f}    '
                         'mse_bt: {:>10.4f}  '
                .format(
                epoch, opt.niter,
                *x_range,
                kl_stats['total_bpd_b'].item(),
                kl_stats['terms_bpd'].item(), kl_stats['prior_bpd_b'].item(), kl_stats['mse_bt'].item()
            ))



        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            logger.info('Generation: eval')

            model.eval()

            with torch.no_grad():

                x_gen_eval = model.gen_samples(
                    sv_x_list[0][:,:,:opt.svpoints], sv_x_list[0][:,:,opt.svpoints:].shape, sv_x_list[0].device, clip_denoised=False).detach().cpu()
                # _20240508添加
                x_gen_eval = torch.cat([x_gen_eval[:, :, :opt.svpoints],
                                        x_gen_eval[:, :, (num_views + MDM_views) * opt.svpoints:]],dim=-1)

                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                logger.info('      [{:>3d}/{:>3d}]  '
                             'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                             'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                    .format(
                    epoch, opt.niter,
                    *gen_eval_range, *gen_stats,
                ))

            export_to_pc_batch('%s/epoch_%03d_samples_eval' % (outf_syn, epoch),
                                    (x_gen_eval.transpose(1, 2)).numpy()*3)

            export_to_pc_batch('%s/epoch_%03d_ground_truth' % (outf_syn, epoch),
                               (sv_x_list[0].transpose(1, 2).detach().cpu()).numpy()*3)

            export_to_pc_batch('%s/epoch_%03d_partial' % (outf_syn, epoch),
                               (sv_x_list[0][:,:,:opt.svpoints].transpose(1, 2).detach().cpu()).numpy()*3)



            model.train()



        if (epoch + 1) % opt.saveIter == 0:

            if should_diag:


                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }

                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))


            if opt.distribution_type == 'multi':
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])

    print('finished')
    dist.destroy_process_group()
    return

def main():
    opt = parse_args()

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)

    ''' workaround '''

    train_dataset, _ = get_dataset(opt)

    #print('train_dataset', train_dataset)#<datasets.shapenet.ShapeNet object at 0x71294c4c4e10>
    #print('len(train_dataset)', len(train_dataset))#1689


    #noises_init = torch.randn(len(train_dataset), opt.npoints - opt.svpoints + 3 * opt.svpoints, opt.nc)
    noises_init = torch.randn(len(train_dataset), opt.npoints-opt.svpoints, opt.nc)
    #_调试代码
    '''
    print('noises_init___')
    print(noises_init.shape)
    print(len(train_dataset))
    print(opt.npoints-opt.svpoints)#1848
    print(opt.nc)
    '''

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot_pc', default='/root/ShapeNetCore.v2.PC15k/')
    # parser.add_argument('--dataroot_sv', default='/root/shapenet_cars_chairs_planes_20views/')
    parser.add_argument('--category', type=str, default='chair')

    parser.add_argument('--num_views', type=int, default= 3 )
    # MDM control, if you want to disable the MDM, set the MDM_views MDM_multiplier latent_multiplier to 0
    parser.add_argument('--MDM_views', type=int, default=3)
    parser.add_argument('--MDM_multiplier', type=float, default=0.25)
    parser.add_argument('--latent_multiplier', type=int, default=20)

    parser.add_argument('--test_type', default='completion')

    parser.add_argument('--bs', type=int, default=50, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=3200, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--svpoints', type=int, default=200)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    '''
    parser.add_argument('--saveIter', default=100, help='unit: epoch')
    parser.add_argument('--diagIter', default=50, help='unit: epoch')
    parser.add_argument('--vizIter', default=50, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, help='unit: iter')
    '''
    parser.add_argument('--saveIter', default=200, type=int, help='unit: epoch')
    parser.add_argument('--diagIter', default=200, type=int, help='unit: epoch')
    parser.add_argument('--vizIter', default=200, type=int, help='unit: epoch')
    parser.add_argument('--print_freq', default=200, type=int, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()
