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
from dataset import People_Dataset, Inference_Dataset, Car_Pseudo_Label_Dataset
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from build_model import ST_CLIP_ViT_cl, CLIP_ViT_cl, CLIP_ConvNext_Car
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

def make_pseudo_label(model, valid_loader):
    model.eval()
    bar = tqdm(valid_loader)
    preds = []
    with torch.no_grad():
        for images in bar:
            images = images.cuda(non_blocking=True)
            pred = model(images).softmax(dim=1).detach().cpu()
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds

def get_class_center_vectors(model, train_loader, class_):
    model.eval()
    bar = tqdm(train_loader)
    preds_all = []
    labels_all = []
    class_center_vector = []
    with torch.no_grad():
        for (images, labels) in bar:
            images = images.cuda(non_blocking=True)
            pred = model(images).softmax(dim=1).detach().cpu()
            preds_all.append(pred)
            labels_all.append(labels)
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    for i in range(class_):
        class_center_vector.append((preds_all[labels_all==i]).mean(dim=0, keepdim=True))
    class_center_vector = torch.cat(class_center_vector, dim=0)
    return class_center_vector.T

def main(configs):
    df = pd.read_csv(args.csv_dir)
    # df = df.sample(1000)
    classes = [6, 11, 65]
    weights_path = [args.model_paths1, args.model_paths2, args.model_paths3]
    pseudo_columns = ['type_pseudo_logits', 'color_pseudo_logits', 'brand_pseudo_logits']
    df[pseudo_columns] = np.nan
    for cl, wps, task, pseudo in zip(classes, weights_path, ['type', 'color', 'brand'], pseudo_columns):
        pred_ = 0
        for wp, cf, mw in zip(wps, configs, args.model_weights):
            dataset = Car_Pseudo_Label_Dataset(df, task, cf.MODEL.img_size)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                shuffle=False, pin_memory=True)
            cf.defrost()
            cf.MODEL.num_classes = cl
            cf.freeze()
            if 'convnext'in cf.MODEL.backbone.model_name:
                model = CLIP_ConvNext_Car(cf, logger)
            elif cf.MODEL.mode == 'st':
                model = ST_CLIP_ViT_cl(cf, logger)
            else:
                model = CLIP_ViT_cl(cf, logger)
            dicts = torch.load(wp)['state_dict']
            model.load_state_dict(dicts, strict=True)
            model.cuda()
            pred = make_pseudo_label(model, data_loader)
            pred_ += pred*mw
        df.loc[df[task].isna(), pseudo] = (pred_.max(dim=1)[0]).numpy()
        df.loc[df[task].isna(), task] = pred_.argmax(dim=1).numpy()
    df.to_csv('data/train_val_cars_pseudo.csv', index=False)
    logger.info('Making pseudo label compelete!')

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

