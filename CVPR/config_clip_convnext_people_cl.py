import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.FOLD = 1
_C.SC = 0.01 #sl_coefficient
_C.SEED = 1
_C.init_lr = 3e-5
_C.batch_size = 128
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# num_car_type: 6
# num_car_colors: 11
# num_car_brands: 65
# people: 30
_C.MODEL = CN()
_C.MODEL.img_size = 224
_C.MODEL.num_classes = 2
_C.MODEL.mode = 'linear'
_C.MODEL.finetune = None
_C.MODEL.output_dir = 'output/people/conv'

# autodl-tmp/pretrained_models/convnext_xxlarge-laion2b_s34b_b82k_augreg_soup.pth
# autodl-tmp/pretrained_models/convnext_xxlarge-laion2b_s34b_b82k_augreg_soup_feature_only.pth

_C.MODEL.backbone = CN()
_C.MODEL.backbone.model_name = 'convnext_xxlarge'
_C.MODEL.backbone.model_path = 'pretrained_weights/convnext_xxlarge-laion2b_s34b_b82k_augreg_soup.pth'
_C.MODEL.backbone.out_dim = 3072
_C.MODEL.backbone.embed_dim = 1024
_C.MODEL.backbone.timm_model_pretrained = False
_C.MODEL.backbone.timm_pool = ''
_C.MODEL.backbone.timm_proj = 'linear'
_C.MODEL.backbone.timm_proj_bias = False
_C.MODEL.backbone.timm_drop = 0.0
_C.MODEL.backbone.timm_drop_path = 0.1

_C.MODEL.backbone.out_indices=[2, 3]
_C.MODEL.backbone.reduced_dim = 128
_C.MODEL.backbone.checkpoint = False
_C.MODEL.backbone.frozen = True
_C.MODEL.backbone.unfreeze_start_stage = 20
_C.MODEL.backbone.unfreeze_stride = 2
_C.MODEL.backbone.min_unfreeze_stage = 20
_C.MODEL.backbone.checkpoint_start_stage = 20
_C.MODEL.backbone.checkpoint_stride = 6

_C.MODEL.ST = CN()
_C.MODEL.ST.centers = 666
_C.MODEL.ST.layers = 2
_C.MODEL.ST.checkpoint = [False, 2]

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'AdamW'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 6e-5
_C.Optimizer.weight_stick_max = 1e-4
_C.Optimizer.weight_stick_min = 1e-5
_C.Optimizer.stick_pow = 1.0

#Loss
_C.Loss = CN()
# ce_loss
# arcface_adaptive_loss
_C.Loss = CN()
_C.Loss.name = 'one_hot_ce'
_C.Loss.s = 30
_C.Loss.m = 0.5

def get_config():
    config = _C.clone()
    return config