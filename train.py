import os
import numpy as np
import argparse
import errno
import math
import pickle
import datetime
import tensorboardX
import torch.distributed
from tqdm import tqdm
import time
import copy
import random
import prettytable
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M  
from lib.model.loss import *
import logger
from logger import colorlogger
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    log.info(f'Saving checkpoint to{chk_path}')
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
def evaluate(args, model_pos, test_loader, datareader):
    log.info('INFO: Testing')
    results_all = []
    model_pos.eval()            
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0

            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    log.info(len(results_all))# 2228
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)# [n_clips, -1, 17, 3]
    log.info(results_all.shape)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])#['s_09_act_02_subact_01_ca_01',...]
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])#[4.656527,...,2.9163694]
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    log.info(f"num_test_frames:{num_test_frames}")# num_test_frames:566920
    frames = np.array(range(num_test_frames))
    # print(split_id_test)
    """ 
    [range(0, 243), range(243, 486), range(486, 729),...,range(566532, 566775)]
    """
    log.info(len(split_id_test))
    action_clips = actions[split_id_test]# ndarray (2228,243)
    factor_clips = factors[split_id_test]# ndarray (2228,243)
    source_clips = sources[split_id_test]# ndarray (2228,243)
    frame_clips = frames[split_id_test]# ndarray (2228,243) [[0,...,242],...,[566532,...,566774]]
    gt_clips = gts[split_id_test]# ndarray (2228,243,17,3)
    assert len(results_all)==len(action_clips)
    
    e1_all = np.zeros(num_test_frames)# ndarray (566920,)
    e2_all = np.zeros(num_test_frames)# ndarray (566920,)
    oc = np.zeros(num_test_frames)# ndarray (566920,)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    #['Direction', 'Discuss', 'Eating', 'Greet', 'Phone', 'Photo', 'Pose', 'Purchase', 'Sitting', 'SittingDown', 'Smoke', 'Wait', 'Walk', 'WalkDog', 'WalkTwo']
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02', 
                  's_09_act_10_subact_02', 
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6] # s_09_act_05_subact_02
        if source in block_list:
            continue
        frame_list = frame_clips[idx] # [0,...,242]
        action = action_clips[idx][0] # Direction
        factor = factor_clips[idx][:,None,None] # ndarray (243,1,1)
        gt = gt_clips[idx] # ndarray (243,17,3)
        pred = results_all[idx] # ndarray (243,17,3)
        pred *= factor# (243,17,3)
        
        # Root-relative Errors
        pred = pred - pred[:,0:1,:] # (243,17,3) 减去Pelvis骨盆的
        gt = gt - gt[:,0:1,:]# (243, 17, 3) 减去Pelvis骨盆的
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    log.info(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    log.info(f'Protocol #1 Error (MPJPE):{e1}mm')
    log.info(f'Protocol #2 Error (P-MPJPE):{e2}mm')
    log.info('----------')
    return e1, e2, results_all
        
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]# (N, T, 17, 2) 
            if not has_3d:
                # 得到2D骨架训练所需要的confidence
                conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
            if args.rootrel: # 相对于根
                batch_gt = batch_gt - batch_gt[:,:,0:1,:]
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)
        
        optimizer.zero_grad()
        if has_3d:
            """ 
            lambda_3d_velocity: 20.0
            lambda_scale: 0.5
            lambda_lv: 0.0
            lambda_lg: 0.0
            lambda_a: 0.0
            lambda_av: 0.0
            """
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt) # 3D
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)# weighted 2D re-projection loss
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt) #LO
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
            loss_3d_w = weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe) # 3D weighted
            
            # Temporal Consistency Loss
            dif_seq = predicted_3d_pos[:,1:,:,:] - predicted_3d_pos[:,:-1,:,:]
            weights_joints = torch.ones_like(dif_seq).cuda()
            weights_mul = w_mpjpe
            assert weights_mul.shape[0] == weights_joints.shape[-2]
            weights_joints = torch.mul(weights_joints.permute(0,1,3,2),weights_mul).permute(0,1,3,2)
            # weights_diff = 0.5
            # index = [1,1,1,1,2,2,2,2,1]
            # dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)), dim=-1)
            loss_diff = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))
            
            loss_total = args.lambda_3d * loss_3d_pos + \
                         args.lambda_scale       * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv          * loss_lv + \
                         args.lambda_lg          * loss_lg + \
                         args.lambda_a           * loss_a  + \
                         args.lambda_av          * loss_av + \
                         args.lambda_3dw          * loss_3d_w + \
                         args.lambda_diff          * loss_diff 
                             
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()
def get_beijing_timestamp():
    local_offset = time.localtime().tm_gmtoff   # 当前机器utc时间偏移量
    beijing_offset = int(8 * 60*60)
    offset = local_offset - beijing_offset
    timestamp = int(datetime.datetime.now().timestamp())
    beijing_timestamp = timestamp - offset

    return beijing_timestamp

def train_with_config(args, opts):

    opts.checkpoint = opts.checkpoint +'_'+ datetime.datetime.fromtimestamp(get_beijing_timestamp()).strftime('%Y_%m_%d_T_%H_%M_%S')
    # global log
    # log = logger.set_save_path(opts.checkpoint)
    # log(args)
    global log
    log = colorlogger(opts.checkpoint, log_name='log.txt')
    log.info(args)
    with open(os.path.join(opts.checkpoint, 'config.yaml'), 'w') as f:
        yaml.dump(args, f, sort_keys=False)
    log.info(f"Number of GPUs found:{torch.cuda.device_count()}")
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    log.info('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)
        
    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = args.data_root, dt_file=args.dt_file)
    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    log.info(f'INFO: Trainable parameter count:{model_params}')

    if torch.cuda.is_available():
        # torch.distributed.init_process_group('nccl', init_method='tcp://localhost:23456', world_size=2, rank=0)
        # model_backbone = nn.parallel.DistributedDataParallel(model_backbone)
        model_backbone = nn.DataParallel(model_backbone)
        # k = torch.cuda.device_count()
        # model_backbone = nn.DataParallel(model_backbone, device_ids=list(range(k)))
        model_backbone = model_backbone.cuda()

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint{chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            log.info(f'Loading checkpoint{chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone            
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint{chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            if args.backbone == 'MotionAGFormer':
                model_backbone.load_state_dict(checkpoint['model'], strict=True)
            else:
                model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone
        
    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:        
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            log.info(f'INFO: Training on {len(train_loader_3d)}(3D)+{len(instav_loader_2d) + len(posetrack_loader_2d)}(2D) batches')
        else:
            log.info(f'INFO: Training on {len(train_loader_3d)}(3D) batches')
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                log.info('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
                
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)
        
        # Training
        for epoch in range(st, args.epochs):
            log.info(f'Training epoch {epoch}.')
            start_time = time.time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            N = 0
                        
            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True) 
            elapsed = (time.time() - start_time) / 60

            if args.no_eval:
                log.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
                log.info('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                log.info(f'Remaining training time: {datetime.timedelta(seconds=time.time() - start_time) * (args.epochs - epoch)}')
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
                
    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)