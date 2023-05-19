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
from utils import *
from dataset import People_Dataset, Inference_Dataset
from build_model import ST_CLIP_ViT_Decoup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--nbatch_log', type=int, default=500)
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def inference_people(model, test_loader):
    model.eval()
    bar = tqdm(test_loader)
    paths = []
    preds = []
    index = [0, 2, 5, 8, 10, 12, 14, 17, 19, 21, 26, 28, 30]
    with torch.no_grad():
        for (images, path) in bar:
            images = images.cuda(non_blocking=True)
            pred = model(images)
            for i in range(len(index)-1):
                pred[i] = pred[i].detach().cpu().softmax(dim=1)
            pred = torch.cat(pred, dim=1)
            paths.extend(path)
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    torch.save(preds, 'data/test/all.pth')
    df = pd.read_csv('data/test/test_code.csv')
    test_codes = df['code'].tolist()
    for i in range(len(test_codes)):
        test_codes[i] = list(map(lambda x:float(x), test_codes[i].strip('[|]').split(' ')))
    test_codes = torch.Tensor(test_codes)

    for i in range(len(test_codes)):
        code = test_codes[i].unsqueeze(0).repeat(len(test_codes), 1)
        dis = ((code-preds)*code)
        dis = 2*dis[:, :8].mean(dim=1)+dis[:, 8:].mean(dim=1)
        path_idx = dis.argsort(dim=0)
        for j in range(10):
            df.loc[i, f'img_{j}'] = paths[path_idx[j]]
    df.to_csv('output/test_people_out.csv', index=False)

def val(model, valid_loader, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_accs = AverageMeter()
    bar = tqdm(valid_loader)
    end = time.time()
    index = [0, 2, 5, 8, 10, 12, 14, 17, 19, 21, 26, 28, 30]
    accs = []
    with torch.no_grad():
        for (images, labels) in bar:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            pred = model(images)
            for i in range(len(index)-1):
                pred[i] = pred[i].softmax(dim=1)
            loss = criterion(pred, labels)
            top1_acc, acc12 = acc_multi_label(pred, labels, index)
            accs.append(acc12)
            losses.update(loss.item(), images.size(0))
            top1_accs.update(top1_acc.item(), images.size(0))
            bar.set_description('Loss_avg: %.5f || top1_acc_avg: %.5f' % (losses.avg, top1_accs.avg))
            batch_time.update(time.time() - end)
            end = time.time()
    accs = torch.Tensor(accs).mean(0)
    if args.local_rank==0:
        logger.info(f"Acc_All: {top1_accs.avg}, Acc_each_cl: {' | '.join([str(float(a)) for a in accs])}")
    return top1_accs.avg, losses.avg

def main(config):
    df = pd.read_csv(args.csv_dir)
    dataset_test = Inference_Dataset('people', config.MODEL.img_size)
    dataset_valid = People_Dataset(df, args.fold, 'valid', config.MODEL.img_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)

    model = ST_CLIP_ViT_Decoup(config, logger)
    dicts = torch.load(args.model_path)['state_dict']
    model.load_state_dict(dicts, strict=True)
    model.cuda()
    criterion = get_criterion_from_config(config)
    inference_people(model, test_loader)
    logger.info('Inference complete!')
    top1_acc, val_loss = val(model, valid_loader, criterion)
    logger.info(f"Loss_val: {val_loss:.5f}, Val_top1_acc: {top1_acc:.5f}")


if __name__ == '__main__':
    args, config = parse_args()
    config.defrost()
    config.MODEL.img_size = args.image_size
    config.local_rank=0
    config.FOLD = args.fold
    config.batch_size = args.batch_size
    config.freeze()
    set_seed(config.SEED)
    os.makedirs(config.MODEL.output_dir, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.output_dir, dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    logger.info(config.dump())
    main(config)

