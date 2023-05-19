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
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from dataset import People_Dataset, Inference_Dataset, People_cl_Dataset
from build_model import ST_CLIP_ViT_cl, CLIP_ViT_cl, CLIP_ConvNext_cl
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-sizes', nargs='+', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config_names', nargs='+', type=str, required=True)
    parser.add_argument('--model-weights', nargs='+', type=float, required=True)
    parser.add_argument('--model_path1', nargs='+', type=str, required=True)
    parser.add_argument('--model_path2', nargs='+', type=str, required=True)
    parser.add_argument('--model_path3', nargs='+', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--nbatch_log', type=int, default=500)
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    configs = [config_from_name(c) for c in args.config_names]
    return args, configs

def inference_people(model, test_loader):
    model.eval()
    bar = tqdm(test_loader)
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
            images = images.cuda(non_blocking=True)
            # preds = (model(images).softmax(dim=1).detach().cpu()+model(images.flip(-1)).softmax(dim=1).detach().cpu())/2
            preds = model(images).softmax(dim=1).detach().cpu()
            preds_all.append(preds)
            labels_all.append(labels)
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return preds_all, labels_all

def main(configs):
    df = pd.read_csv(args.csv_dir)
    # df = df.sample(n=100)
    classes = [2, 3, 3, 2, 2, 2, 3, 2, 2, 5, 2, 2]
    tags = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    new_model_paths = []
    if args.model_path3 is not None:
        for w1, w2, w3 in zip(args.model_path1, args.model_path2, args.model_path3):
            new_model_paths.append([w1, w2, w3])
    else:
        for w1, w2 in zip(args.model_path1, args.model_path2):
            new_model_paths.append([w1, w2])
    preds_val, labels_val = [], []
    for c, t, nmps in zip(classes, tags, new_model_paths):
        pred_val = 0
        for nmp, mw, cfg in zip(nmps, args.model_weights, configs):
            cfg.defrost()
            cfg.MODEL.num_classes = c
            cfg.freeze()
            dataset_valid = People_cl_Dataset(df, args.fold, 'valid', t, cfg.MODEL.img_size)
            valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                                shuffle=False, pin_memory=True)
            if 'convnext'in cfg.MODEL.backbone.model_name:
                    model = CLIP_ConvNext_cl(cfg, logger)
            elif cfg.MODEL.mode == 'st':
                model = ST_CLIP_ViT_cl(cfg, logger)
            else:
                model = CLIP_ViT_cl(cfg, logger)
            dicts = torch.load(nmp)['state_dict']
            model.load_state_dict(dicts, strict=True)
            model.cuda()
            pred_v, label_val = val(model, valid_loader)
            pred_val = pred_val + mw*pred_v

        pred_val = F.one_hot(pred_val.argmax(dim=1), pred_val.shape[-1])
        preds_val.append(pred_val)
        labels_val.append(label_val)

    preds_val = torch.cat(preds_val, dim=1)
    labels_val = torch.cat(labels_val, dim=1)
    eq_num = (preds_val.eq(labels_val).sum(dim=1)==preds_val.shape[-1]).sum()
    acc = eq_num/preds_val.shape[0]
    logger.info(f'Val acc: {acc*100:.5f}')


if __name__ == '__main__':
    args, configs = parse_args()
    for config, img_size in zip(configs, args.image_sizes):
        config.defrost()
        config.MODEL.img_size = img_size
        if img_size == 336:
            config.MODEL.backbone.model_path = 'pretrained_weights/EVA02_CLIP_L_336_psz14_s6B_vision.pt'
        config.local_rank=0
        config.batch_size = args.batch_size
        config.freeze()
        set_seed(config.SEED)
    os.makedirs('output/people_inference', exist_ok=True)
    logger = create_logger(output_dir='output/people_inference', dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    logger.info(config.dump())
    main(configs)

