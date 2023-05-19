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
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument('--model_path', nargs='+', type=str, required=True)
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
    with torch.no_grad():
        for (images, path) in bar:
            images = images.cuda(non_blocking=True)
            # pred = (model(images).softmax(dim=1).detach().cpu()+model(images.flip(-1)).softmax(dim=1).detach().cpu())/2
            pred = model(images).softmax(dim=1).detach().cpu()
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

def main(config):
    df = pd.read_csv(args.csv_dir)
    # df = df.sample(n=100)
    classes = [2, 3, 3, 2, 2, 2, 3, 2, 2, 5, 2, 2]
    tags = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    preds_val, labels_val = [], []
    for c, t, mp in zip(classes, tags, args.model_path):
        config.defrost()
        config.MODEL.num_classes = c
        config.freeze()
        dataset_valid = People_cl_Dataset(df, args.fold, 'valid', t, config.MODEL.img_size)
        valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)
        if 'convnext'in config.MODEL.backbone.model_name:
                model = CLIP_ConvNext_cl(config, logger)
        elif config.MODEL.mode == 'st':
            model = ST_CLIP_ViT_cl(config, logger)
        else:
            model = CLIP_ViT_cl(config, logger)
        dicts = torch.load(mp)['state_dict']
        model.load_state_dict(dicts, strict=True)
        model.cuda()
        pred_val, label_val = val(model, valid_loader)
        pred_val = F.one_hot(pred_val.argmax(dim=1), pred_val.shape[-1])
        preds_val.append(pred_val)
        labels_val.append(label_val)

    preds_val = torch.cat(preds_val, dim=1)
    labels_val = torch.cat(labels_val, dim=1)
    eq_num = (preds_val.eq(labels_val).sum(dim=1)==preds_val.shape[-1]).sum()
    acc = eq_num/preds_val.shape[0]
    logger.info(f'Val acc: {acc*100:.5f}')


if __name__ == '__main__':
    args, config = parse_args()
    config.defrost()
    config.MODEL.img_size = args.image_size
    if args.image_size == 336:
            config.MODEL.backbone.model_path = 'pretrained_weights/EVA02_CLIP_L_336_psz14_s6B_vision.pt'
    config.local_rank=0
    config.FOLD = args.fold
    config.batch_size = args.batch_size
    config.freeze()
    set_seed(config.SEED)
    os.makedirs('output/people_inference', exist_ok=True)
    logger = create_logger(output_dir='output/people_inference', dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    logger.info(config.dump())
    main(config)

