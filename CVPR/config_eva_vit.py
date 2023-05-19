import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.FOLD = 2
_C.SC = 1. #sl_coefficient
_C.SEED = 1
_C.init_lr = 3e-5
_C.batch_size = 128
_C.index = [0, 2, 5, 8, 10, 12, 14, 17, 19, 21, 26, 28, 30]
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# num_car_type: 6
# num_car_colors: 11
# num_car_brands: 65
# people: 30
_C.MODEL = CN()
_C.MODEL.img_size = 224
_C.MODEL.num_classes = [2, 3, 3, 2, 2, 2, 3, 2, 2, 5, 2, 2]
_C.MODEL.mode = 'Single'
_C.MODEL.finetune = None
_C.MODEL.output_dir = 'output/people/eva-l-people-all'
# pretrained_weights/EVA02_CLIP_L_336_psz14_s6B_vision_decoupler.pt
# pretrained_weights/EVA02_CLIP_L_psz14_s4B_vision_decoupler.pt

_C.MODEL.type = 'fp32'
_C.MODEL.backbone = CN()
_C.MODEL.backbone.model_name = 'eva-decoup-l'
_C.MODEL.backbone.model_path = 'pretrained_weights/EVA02_CLIP_L_336_psz14_s6B_vision_decoupler.pt'
_C.MODEL.backbone.patch_size = 14
_C.MODEL.backbone.output_dim = 768
_C.MODEL.backbone.use_mean_pooling = False
_C.MODEL.backbone.init_values = None
_C.MODEL.backbone.patch_dropout = 0.
_C.MODEL.backbone.width = 1024
_C.MODEL.backbone.depth = 24
_C.MODEL.backbone.num_heads = 16
_C.MODEL.backbone.mlp_ratio = 2.6667
_C.MODEL.backbone.qkv_bias = True
_C.MODEL.backbone.drop_path_rate = 0.
_C.MODEL.backbone.xattn = True
_C.MODEL.backbone.rope = True
_C.MODEL.backbone.postnorm = False
_C.MODEL.backbone.pt_hw_seq_len = 16
_C.MODEL.backbone.intp_freq = True
_C.MODEL.backbone.naiveswiglu = True
_C.MODEL.backbone.subln = True

_C.MODEL.backbone.reduced_dim = 128
_C.MODEL.backbone.num_stage = 3
_C.MODEL.backbone.checkpoint = True
_C.MODEL.backbone.frozen = True
_C.MODEL.backbone.unfreeze_start_stage = 16
_C.MODEL.backbone.unfreeze_stride = 1
_C.MODEL.backbone.min_unfreeze_stage = 18
_C.MODEL.backbone.checkpoint_start_stage = 17
_C.MODEL.backbone.checkpoint_stride = 20
_C.MODEL.backbone.shared_dpth = 23
_C.MODEL.backbone.indepdent_depth = 1
_C.MODEL.backbone.num_decouplers = 12

#eva-gaint
# _C.MODEL.type = 'fp32'
# _C.MODEL.output_dir = 'autodl-tmp/output/eva-g'
# _C.MODEL.backbone = CN()
# _C.MODEL.backbone.model_name = 'eva-clip-g-14-x'
# _C.MODEL.backbone.model_path = 'autodl-tmp/pretrained_models/EVA01_CLIP_g_14_plus_psz14_s11B_vision.pt'
# _C.MODEL.backbone.patch_size = 14
# _C.MODEL.backbone.output_dim = 1024
# _C.MODEL.backbone.use_mean_pooling = False
# _C.MODEL.backbone.init_values = None
# _C.MODEL.backbone.patch_dropout = 0.
# _C.MODEL.backbone.width = 1408
# _C.MODEL.backbone.depth = 40
# _C.MODEL.backbone.num_heads = 16
# _C.MODEL.backbone.mlp_ratio = 4.3637
# _C.MODEL.backbone.qkv_bias = True
# _C.MODEL.backbone.drop_path_rate = 0.
# _C.MODEL.backbone.xattn = True
# _C.MODEL.backbone.rope = False
# _C.MODEL.backbone.postnorm = False
# _C.MODEL.backbone.pt_hw_seq_len = 16
# _C.MODEL.backbone.intp_freq = False
# _C.MODEL.backbone.naiveswiglu = False
# _C.MODEL.backbone.subln = False

# _C.MODEL.backbone.reduced_dim = 128
# _C.MODEL.backbone.num_stage = 4
# _C.MODEL.backbone.checkpoint = False
# _C.MODEL.backbone.frozen = True
# _C.MODEL.backbone.unfreeze_start_stage = '30'

#eva-E
# _C.MODEL.type = 'fp32'
# _C.MODEL.output_dir = 'autodl-tmp/output/eva-e'
# _C.MODEL.backbone = CN()
# _C.MODEL.backbone.model_name = 'eva-clip-e'
# _C.MODEL.backbone.model_path = 'autodl-tmp/pretrained_models/EVA02_CLIP_E_psz14_plus_s9B_vision.pt'
# _C.MODEL.backbone.patch_size = 14
# _C.MODEL.backbone.output_dim = 1024
# _C.MODEL.backbone.use_mean_pooling = False
# _C.MODEL.backbone.init_values = None
# _C.MODEL.backbone.patch_dropout = 0.
# _C.MODEL.backbone.width = 1792
# _C.MODEL.backbone.depth = 64
# _C.MODEL.backbone.num_heads = 16
# _C.MODEL.backbone.mlp_ratio = 8.571428571428571
# _C.MODEL.backbone.qkv_bias = True
# _C.MODEL.backbone.drop_path_rate = 0.
# _C.MODEL.backbone.xattn = True
# _C.MODEL.backbone.rope = False
# _C.MODEL.backbone.postnorm = True
# _C.MODEL.backbone.pt_hw_seq_len = 16
# _C.MODEL.backbone.intp_freq = False
# _C.MODEL.backbone.naiveswiglu = False
# _C.MODEL.backbone.subln = False

# _C.MODEL.backbone.reduced_dim = 128
# _C.MODEL.backbone.num_stage = 3
# _C.MODEL.backbone.checkpoint = False
# _C.MODEL.backbone.frozen = True
# _C.MODEL.backbone.unfreeze_start_stage = '54'

_C.MODEL.ST = CN()
_C.MODEL.ST.centers = 30
_C.MODEL.ST.layers = 2
_C.MODEL.ST.mlp_ratio = 4.0
_C.MODEL.ST.checkpoint = [False, 2]

_C.MODEL.backbone.VPT = CN()
_C.MODEL.backbone.VPT.vpt_type = 'deep'
_C.MODEL.backbone.VPT.p_num = 60

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'AdamS'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 3e-5
_C.Optimizer.weight_stick_max = 5e-5
_C.Optimizer.weight_stick_min = 1e-5
_C.Optimizer.stick_pow = 1.0
#Loss
# arcface_loss
# multi_ce
_C.Loss = CN()
_C.Loss.name = 'multi_ce'
_C.Loss.s = 20
_C.Loss.m = 0.2

def get_config():
    config = _C.clone()
    return config