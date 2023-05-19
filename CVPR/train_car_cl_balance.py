import os
import time
import argparse
import datetime
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from logger import create_logger
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils import *
from dataset import Car_cl_balance_Dataset
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from build_model import ST_Net_TIMM, CLIP_ViT_cl, ST_CLIP_ViT_cl, CLIP_ConvNext_Car

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nbatch_log', type=int, default=500)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def train_epoch(cur_epoch, model, train_loader, optimizer, criterion, scaler, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    bar = tqdm(train_loader)
    steps = 0
    for (images, labels) in bar:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        if cur_epoch<=args.warmup_epochs:
            lr = get_warm_up_lr(cur_epoch, steps, args.warmup_epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        else:
            lr = get_train_epoch_lr(cur_epoch, steps, args.epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        with autocast():
            preds = model(images)
            loss = criterion(preds, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        reduced_loss = reduce_tensor(loss.data)
        losses.update(reduced_loss.item(), images.size(0))
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        bar.set_description('Lr: %.7f, L_cur: %.4f, L_avg: %.4f' % (lr, losses.val, losses.avg))
        if batch_time.count%args.nbatch_log==0 and args.local_rank==0:
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info('Epoch: %d || Iter: [%d/%d], Lr: %.7f, Memory_used: %.0fMB, Loss_cur: %.5f, Loss_avg: %.5f, \
                        Time_avg: %.3f, Time_total: %.3f' % (cur_epoch, batch_time.count, len(train_loader), lr, mu, losses.val, losses.avg, batch_time.avg, batch_time.sum))
        steps += 1
    return losses.avg

def val_epoch(model, valid_loader, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_accs = AverageMeter()
    bar = tqdm(valid_loader)
    end = time.time()
    with torch.no_grad():
        for (images, labels) in bar:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            preds = model(images)
            loss = criterion(preds, labels)
            top1_acc = accuracy(preds, labels)[0]
            reduced_loss = reduce_tensor(loss)
            reduced_top1_acc = reduce_tensor(top1_acc)
            losses.update(reduced_loss.item(), images.size(0))
            top1_accs.update(reduced_top1_acc.item(), images.size(0))
            bar.set_description('Loss_avg: %.5f || top1_acc_avg: %.5f' % (losses.avg, top1_accs.avg))
            batch_time.update(time.time() - end)
            end = time.time()
    return top1_accs.avg, losses.avg

def main(config):
    df = pd.read_csv(args.csv_dir)
    # df = df.sample(1000)
    dataset_train = Car_cl_balance_Dataset(df, args.fold, 'train', config.tag, config.MODEL.img_size)
    dataset_train.update()
    dataset_valid = Car_cl_balance_Dataset(df, args.fold, 'valid', config.tag, config.MODEL.img_size)
    train_sampler = DistributedSampler(dataset_train)
    valid_sampler = DistributedSampler(dataset_valid)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True, sampler=valid_sampler, drop_last=True)
    # CLIP_ViT_cl
    # ST_CLIP_ViT_cl
    if 'convnext'in config.MODEL.backbone.model_name:
        model = CLIP_ConvNext_Car(config, logger)
    elif config.MODEL.mode == 'st':
        model = ST_CLIP_ViT_cl(config, logger)
    else:
        model = CLIP_ViT_cl(config, logger)
    if config.MODEL.finetune != None:
        load_ckpt_finetune(config.MODEL.finetune, model, logger=logger, args=args)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None, find_unused_parameters=True) #find_unused_parameters=True
    # model._set_static_graph()
    model = torch.compile(model)
    optimizer = get_optim_from_config(config, model, 'embed')

    criterion = get_criterion_from_config(config)

    scaler = GradScaler()
    # strat training
    if args.local_rank==0:
        logger.info(f"Start training in mode '{config.MODEL.mode}'")
    start_time = time.time()
    best_top1_acc = -1
    args.epochs += 1
    unfreeze_stage = config.MODEL.backbone.unfreeze_start_stage
    checkpoint_start_stage = config.MODEL.backbone.checkpoint_start_stage
    for epoch in range(1, args.epochs):
        if epoch>6:
            break
        if args.local_rank==0:
            logger.info(f"----------[Epoch {epoch}]----------")
            logger.info(f'unfreeze start stage:{unfreeze_stage} || checkpoint stage:{checkpoint_start_stage}')

        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, scaler, args)
        top1_acc, val_loss = val_epoch(model, valid_loader, criterion)
        if epoch%4==0:
            dataset_train.update()
            train_sampler = DistributedSampler(dataset_train)
            train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
        if epoch%2==0:
            unfreeze_stage = min(config.MODEL.backbone.min_unfreeze_stage, unfreeze_stage+config.MODEL.backbone.unfreeze_stride)
            if config.MODEL.backbone.checkpoint == True:
                checkpoint_start_stage += config.MODEL.backbone.checkpoint_stride
                model.module.backbone.checkpoint_start_stage = checkpoint_start_stage
            unfreeze = False
            for name, param in model.module.backbone.named_parameters():
                if str(unfreeze_stage) in name:
                    unfreeze = True
                param.requires_grad=unfreeze

        best_top1_acc = -1
        if args.local_rank==0:
            logger.info(f"Epoch: {epoch} || Loss_train: {train_loss:.5f}, Loss_val: {val_loss:.5f}, Val_top1_acc: {top1_acc:.5f}")
            if top1_acc > best_top1_acc and epoch>=3:
                best_top1_acc = top1_acc
                save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_best_ep{epoch}.pth')
                save_checkpoint(model, save_path)
                logger.info(f"Save best model to {save_path}, with best top1_acc: {best_top1_acc}")
            logger.info(f'Epoch {epoch} time cost: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
        if config.Loss.name == 'adaptive_arcface_loss':
            criterion.update(epoch, args.epochs-1)
    # finish training
    # if args.local_rank==0:
    #     save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_last.pth')
    #     save_checkpoint(model, save_path)
    #     logger.info(f"Save last model to {save_path}")
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     logger.info('Training time in total: {}'.format(total_time_str))

if __name__ == '__main__':
    args, config = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    config.defrost()
    config.MODEL.img_size = args.image_size
    config.FOLD = args.fold
    config.init_lr = args.init_lr
    config.batch_size = args.batch_size
    config.local_rank = args.local_rank
    config.world_size = args.world_size
    if 'type' in args.csv_dir:
        config.MODEL.num_classes = 6
        config.MODEL.output_dir = config.MODEL.output_dir+'-car-type'
        config.tag = 'type'
    elif 'color' in args.csv_dir:
        config.MODEL.num_classes = 11
        config.MODEL.output_dir = config.MODEL.output_dir+'-car-color'
        config.tag = 'color'
    elif 'brand' in args.csv_dir:
        config.MODEL.num_classes = 65
        config.MODEL.output_dir = config.MODEL.output_dir+'-car-brand'
        config.tag = 'brand'
    config.freeze()

    set_seed(config.SEED)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    os.makedirs(config.MODEL.output_dir, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.output_dir, dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    logger.info(config.dump())
    main(config)

