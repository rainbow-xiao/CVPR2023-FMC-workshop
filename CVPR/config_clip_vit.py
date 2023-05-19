import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.FOLD = 1
_C.SC = 1 #sl_coefficient
_C.SEED = 1
_C.init_lr = 3e-5
_C.batch_size = 128
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.img_size = 224
_C.MODEL.num_classes = 30
_C.MODEL.f_center = 12
_C.MODEL.finetune = None

# vit_large_patch14_224_clip_laion2b
# vit_huge_patch14_224_clip_laion2b

# _C.MODEL.output_dir = 'autodl-tmp/output/ViT-large'
# _C.MODEL.backbone = CN()
# _C.MODEL.backbone.model_name = 'ViT-L-14'
# _C.MODEL.backbone.model_path = 'autodl-tmp/pretrained_models/ViT-L-14-laion2b_s32b_b82k.pth'
# _C.MODEL.backbone.patch_size = 14
# _C.MODEL.backbone.width = 1024
# _C.MODEL.backbone.layers = 24
# _C.MODEL.backbone.heads = 16
# _C.MODEL.backbone.mlp_ratio = 4.0
# _C.MODEL.backbone.global_average_pool = False
# _C.MODEL.backbone.attentional_pool = False
# _C.MODEL.backbone.attn_pooler_heads = 8
# _C.MODEL.backbone.n_queries = 256
# _C.MODEL.backbone.reduced_dim = 128
# _C.MODEL.backbone.output_dim = 768
# _C.MODEL.backbone.num_stage = 4
# _C.MODEL.backbone.checkpoint = False
# _C.MODEL.backbone.frozen = True
# _C.MODEL.backbone.unfreeze_start_stage = '24'

_C.MODEL.output_dir = 'autodl-tmp/output/ViT-huge'
_C.MODEL.backbone = CN()
_C.MODEL.backbone.model_name = 'ViT_huge'
_C.MODEL.backbone.model_path = 'autodl-tmp/pretrained_models/ViT-H-14.pt'
_C.MODEL.backbone.patch_size = 14
_C.MODEL.backbone.width = 1280
_C.MODEL.backbone.layers = 32
_C.MODEL.backbone.heads = 16
_C.MODEL.backbone.mlp_ratio = 4.0
_C.MODEL.backbone.global_average_pool = False
_C.MODEL.backbone.attentional_pool = False
_C.MODEL.backbone.attn_pooler_heads = 8
_C.MODEL.backbone.n_queries = 256
_C.MODEL.backbone.reduced_dim = 128
_C.MODEL.backbone.output_dim = 1024

_C.MODEL.backbone.num_stage = 4
_C.MODEL.backbone.checkpoint = False
_C.MODEL.backbone.frozen = True
_C.MODEL.backbone.unfreeze_start_stage = '24'

# _C.MODEL.output_dir = 'autodl-tmp/output/ViT-bigG'
# _C.MODEL.backbone = CN()
# _C.MODEL.backbone.model_name = 'ViT-bigG'
# _C.MODEL.backbone.model_path = 'autodl-tmp/pretrained_models/ViT-bigG-14-vision-laion2b_s39b_b160k.pth'
# _C.MODEL.backbone.patch_size = 14
# _C.MODEL.backbone.width = 1664
# _C.MODEL.backbone.layers = 48
# _C.MODEL.backbone.heads = 16
# _C.MODEL.backbone.mlp_ratio = 4.9231
# _C.MODEL.backbone.global_average_pool = False
# _C.MODEL.backbone.attentional_pool = False
# _C.MODEL.backbone.attn_pooler_heads = 8
# _C.MODEL.backbone.n_queries = 256
# _C.MODEL.backbone.reduced_dim = 128
# _C.MODEL.backbone.output_dim = 1280
# _C.MODEL.backbone.checkpoint = False
# _C.MODEL.backbone.frozen = True
# _C.MODEL.backbone.unfreeze_start_stage = '24'

_C.MODEL.ST = CN()
_C.MODEL.ST.centers = 384
_C.MODEL.ST.layers = 1
_C.MODEL.ST.mlp_ratio = 4.0
_C.MODEL.ST.checkpoint = [False, 1]

_C.MODEL.backbone.VPT = CN()
_C.MODEL.backbone.VPT.vpt_type = 'deep'
_C.MODEL.backbone.VPT.p_num = 60

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'AdamW'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-4

#Loss
_C.Loss = CN()
_C.Loss.name = 'cos_loss'

def get_config():
    config = _C.clone()
    return config