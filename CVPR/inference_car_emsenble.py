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
from dataset import People_Dataset, Inference_Dataset, Car_cl_Dataset
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from build_model import ST_CLIP_ViT_cl, CLIP_ViT_cl_car, CLIP_ConvNext_cl
from tqdm import trange
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-sizes', nargs='+', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config_names', nargs='+', type=str, required=True)
    parser.add_argument('--model-weights', nargs='+', type=float, required=True)
    parser.add_argument('--model_paths1', nargs='+', type=str, required=True)
    parser.add_argument('--model_paths2', nargs='+', type=str, required=True)
    parser.add_argument('--model_paths3', nargs='+', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    configs = [config_from_name(c) for c in args.config_names]
    return args, configs

def inference_car(model, data_loader):
    model.eval()
    bar = tqdm(data_loader)
    paths = []
    preds = []
    with torch.no_grad():
        for (images, path) in bar:
            images = images.cuda(non_blocking=True)
            pred = (model(images).softmax(dim=1).detach().cpu()+model(images.flip(-1)).softmax(dim=1).detach().cpu())/2
            # pred = model(images).softmax(dim=1).detach().cpu()
            paths.extend(path)
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds, paths

def val(model, valid_loader):
    model.eval()
    bar = tqdm(valid_loader)
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for (images, labels) in bar:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            preds = model(images)
            preds_all.append(preds)
            labels_all.append(labels)
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return preds_all, labels_all

def main(configs):
    df = pd.read_csv(args.csv_dir)
    classes = [6, 11, 65]
    weights_path = [args.model_paths1, args.model_paths2, args.model_paths3]
    preds = []
    for cl, wps in zip(classes, weights_path):
        pred_ = 0
        for wp, cf, mw in zip(wps, configs, args.model_weights):
            dataset = Inference_Dataset('car', cf.MODEL.img_size)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                shuffle=False, pin_memory=True)
            cf.defrost()
            cf.MODEL.num_classes = cl
            cf.freeze()
            if 'convnext'in cf.MODEL.backbone.model_name:
                model = CLIP_ConvNext_cl(cf, logger)
            elif cf.MODEL.mode == 'st':
                model = ST_CLIP_ViT_cl(cf, logger)
            else:
                model = CLIP_ViT_cl_car(cf, logger)
            dicts = torch.load(wp)['state_dict']
            model.load_state_dict(dicts, strict=True)
            model.cuda()
            pred, paths = inference_car(model, data_loader)
            pred_ += pred*mw
        preds.append(pred_)
    preds = torch.cat(preds, dim=1)
    # torch.save(preds, f'output/car_emsenble.pth')
    label_1 = torch.tensor(df['type'].tolist())
    label_2 = torch.tensor(df['color'].tolist())
    label_3 = torch.tensor(df['brand'].tolist())
    label_1 = F.one_hot(label_1.long(), 6).float()
    label_2 = F.one_hot(label_2.long(), 11).float()
    label_3 = F.one_hot(label_3.long(), 65).float()
    labels = torch.cat([label_1, label_2, label_3], dim=1)
    for i in range(len(labels)):
        code = labels[i].unsqueeze(0).repeat(len(labels), 1)
        mse = ((code-preds)*code).sum(dim=1)
        path_idx = mse.argsort(dim=0)
        for j in range(10):
            df.loc[i, f'img_{j}'] = paths[path_idx[j]]
    df.to_csv('output/test_car_out.csv', index=False)
    logger.info('Inference complete!')

if __name__ == '__main__':
    args, configs = parse_args()
    for config, img_size in zip(configs, args.image_sizes):
        config.defrost()
        config.MODEL.img_size = img_size
        config.local_rank=0
        config.batch_size = args.batch_size
        config.freeze()
        set_seed(config.SEED)
    os.makedirs('output/car_inference', exist_ok=True)
    logger = create_logger(output_dir='output/car_inference', dist_rank=args.local_rank, name='car_inference')
    logger.info(config.dump())
    main(configs)

