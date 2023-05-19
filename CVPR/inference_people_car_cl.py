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
from dataset import People_Dataset, People_Car_Inference_Dataset, Car_cl_Dataset
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from build_model import ST_CLIP_ViT_cl, CLIP_ViT_cl, CLIP_ConvNext_cl
from tqdm import trange
import torch.nn.functional as F
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def inference(model, test_loader):
    model.eval()
    bar = tqdm(test_loader)
    paths = []
    preds = []
    with torch.no_grad():
        for (images, path) in bar:
            images = images.cuda(non_blocking=True)
            pred = model(images).argmax(dim=1).detach().cpu()
            paths.extend(path)
            preds.append(pred)
    preds = torch.cat(preds, dim=0).numpy()
    return preds, paths

def main(config):
    dataset_test = People_Car_Inference_Dataset(args.test_data_path, config.MODEL.img_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)
    if 'convnext'in config.MODEL.backbone.model_name:
        model = CLIP_ConvNext_cl(config, logger)
    elif config.MODEL.mode == 'st':
        model = ST_CLIP_ViT_cl(config, logger)
    else:
        model = CLIP_ViT_cl(config, logger)
    dicts = torch.load(args.model_path)['state_dict']
    model.load_state_dict(dicts, strict=True)
    model.cuda()
    preds, paths = inference(model, test_loader)
    df = pd.DataFrame({'img_path': paths, 'label': preds}).reset_index(drop=True)
    df.to_csv('output/test_people_car_cl_pred.csv', index=False)
    logger.info('Inference complete!')

if __name__ == '__main__':
    args, config = parse_args()
    config.defrost()
    config.MODEL.img_size = args.image_size
    config.local_rank=0
    config.batch_size = args.batch_size
    config.freeze()
    set_seed(config.SEED)
    os.makedirs('output/car_inference', exist_ok=True)
    logger = create_logger(output_dir='output/car_inference', dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    logger.info(config.dump())
    main(config)


