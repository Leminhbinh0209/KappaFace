import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model
from dataset import MXFaceDataset, SyntheticDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
import wandb
from memory import LatentMemory
from momentum_head import MomentumCalcHead
import random
import numpy as np
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1337)
np.random.seed(1337)
def _init_fn():
    np.random.seed(1337)
def gather(tensor, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
  
    rank = dist.get_rank()
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        dist.gather(tensor, gather_list=tensor_list, group=group)
    else:
        dist.gather(tensor, dst=root, group=group)
def main(args):
    cfg = get_config(args.config)
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    if cfg.rec == "synthetic":
        train_set = SyntheticDataset(local_rank=local_rank)
    else:
        train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank, n_classes=cfg.num_classes)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
    # memory
    SAMPLE_NUMS = train_set.get_sample_num_of_each_class()
    state = None
    
    if rank == 0:
        if not cfg.momentum:
            calc_head = LatentMemory(n_data=train_set.__len__(), feat_dim=512, cls_positive=SAMPLE_NUMS, T=cfg.T, gamma=cfg.gamma).to(local_rank)
        else:
            calc_head = MomentumCalcHead(feat_dim=512, cls_positive=SAMPLE_NUMS, T=cfg.T, gamma=cfg.gamma)
        wandber = wandb.init(config=cfg, project="research-face", name=cfg.instance)
        wandber.watch(backbone)
        
    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")
    if cfg.momentum:
        logging.info("Initializing momentum encoder!")
        momentum_encoder = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
        for param_q, param_k in zip(backbone.parameters(), momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        momentum_encoder.eval()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()
    
    margin_softmax = losses.get_loss(cfg.loss, cfg)
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    num_image = len(train_set)
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    def lr_step_func(current_step):
        cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]
        if current_step < cfg.warmup_step:
            return current_step / cfg.warmup_step
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= current_step])

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=lr_step_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=lr_step_func)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))
    print(len(train_loader))
    val_target = cfg.val_targets
    callback_verification = CallBackVerification(2000, rank, val_target, cfg.rec)
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    
    loss = AverageMeter()
    start_epoch = 0
    global_step = 0
    weights = torch.ones(cfg.num_classes).to(local_rank)
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    best_results = {v:0 for v in val_target}
    if rank == 0:
        for v in val_target:
            if not os.path.exists(cfg.output + '/' + v):
                os.mkdir(cfg.output + '/' + v)
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label, indexes) in enumerate(train_loader):
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc, weights)
            if cfg.fp16:
                features.backward(grad_amp.scale(x_grad))
                grad_amp.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(opt_backbone)
                grad_amp.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, scheduler_backbone.get_last_lr()[0], grad_amp)
            scheduler_backbone.step()
            scheduler_pfc.step()
            # update memory
            if cfg.momentum:
                with torch.no_grad():
                    for param_q, param_k in zip(backbone.parameters(), momentum_encoder.parameters()):
                        param_k.data = param_k.data * 0.9 + param_q.data * (1. - 0.9)
                    features = F.normalize(momentum_encoder(img))
            total_features = torch.zeros(
            size=[cfg.batch_size * world_size, cfg.embedding_size], device=local_rank)
            dist.all_gather(list(total_features.chunk(world_size, dim=0)), features.data)
            
            total_label = torch.zeros(
                size=[cfg.batch_size * world_size], device=local_rank, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(world_size, dim=0)), label)
            
            total_indexes = torch.zeros(
                size=[cfg.batch_size * world_size], device=local_rank, dtype=torch.long)
            dist.all_gather(list(total_indexes.chunk(world_size, dim=0)), indexes)
            
            total_features.requires_grad = False
            if rank == 0:
                calc_head(total_features, total_label, total_indexes)
            
            report = callback_verification(global_step, backbone)
            to_save = []
            if report is not None:
                for dataset, res in report.items():
                    if best_results[dataset] < res:
                        best_results[dataset] = res
                        to_save.append(dataset)
            if to_save.__len__() > 0:
                callback_checkpoint(global_step, backbone, module_partial_fc, to_save, None)
        report = callback_verification(2000, backbone)
        to_save = []
        if report is not None:
            for dataset, res in report.items():
                if best_results[dataset] < res:
                    best_results[dataset] = res
                    to_save.append(dataset)
    
        state = {}
        if rank == 0:
            kappas = calc_head.update_kappa()
            weights = calc_head.update_weights()
            state = {'kappas': kappas, 'samples': SAMPLE_NUMS, 'weights': weights}
            kappa_report = {'max_kappa': kappas.max().item(), 'min_kappa': kappas.min().item(), 'mean_kappa': kappas.mean().item()}
            report = dict({'epoch': epoch, 'train_loss': loss.avg}, **report)
            report = dict(report, **kappa_report)
            wandber.log(report)
        else:
            weights = torch.ones(cfg.num_classes).to(local_rank) * -1.
        #dist.broadcast(weights, src=0)
        total_weights = torch.zeros(
            size=[cfg.num_classes * world_size], device=local_rank)
        dist.all_gather(list(total_weights.chunk(world_size, dim=0)), weights)
        weights = total_weights[total_weights != -1]
        assert weights.__len__() == cfg.num_classes
        callback_checkpoint(global_step, backbone, module_partial_fc, to_save, state)
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch KappaFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
          