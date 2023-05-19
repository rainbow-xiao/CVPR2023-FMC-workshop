import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from utils import *
import timm
from torch.utils.checkpoint import checkpoint
from functools import partial

class CLIP_ConvNext_cl(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.backbone = build_clip(config, logger)
        self.head = nn.Linear(config.MODEL.backbone.out_dim, config.MODEL.num_classes)

        if config.MODEL.backbone.frozen:
            unfreeze = False
            stage = str(config.MODEL.backbone.unfreeze_start_stage-1)
            for name, param in self.backbone.named_parameters():
                if stage in name:
                    unfreeze = True
                param.requires_grad=unfreeze
                if param.requires_grad and config.local_rank==0 and stage in name:
                    logger.info(f"{name} is set to be trainable.")

        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing(config.MODEL.backbone.checkpoint_start_stage, config.MODEL.backbone.checkpoint_start_block)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = x.mean(1)
        x = self.head(x)
        return x

class ST_CLIP_ViT_Decoup(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.backbone = build_clip(config, logger)
        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()

        self.input_size = config.MODEL.img_size
        self.patch_size = config.MODEL.backbone.patch_size
        assert self.input_size%self.patch_size==0

        self.head = nn.ModuleList([nn.Linear(config.MODEL.backbone.output_dim, c) for c in config.MODEL.num_classes])
        # self.head = nn.ModuleList([ArcMarginProduct(config.MODEL.backbone.output_dim, c) for c in config.MODEL.num_classes])
        # self.head = nn.ModuleList([Space_Attention(in_dim=config.MODEL.backbone.output_dim, out_dim=c, centers=c*2) for c in config.MODEL.num_classes])
        if config.MODEL.backbone.frozen:
            unfreeze = False
            stage = str(config.MODEL.backbone.unfreeze_start_stage-1)
            for name, param in self.backbone.named_parameters():
                if stage in name:
                    unfreeze = True
                param.requires_grad=unfreeze
                if param.requires_grad and config.local_rank==0 and stage in name:
                    logger.info(f"{name} is set to be trainable.")

    def forward(self, x):
        x = self.backbone(x)
        xs = []
        for i in range(len(self.head)):
            xs.append(self.head[i](x[i]))
        return xs

class CLIP_ViT_cl_car(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.backbone = build_clip(config, logger)
        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()
        self.head1 = nn.Linear(config.MODEL.backbone.output_dim, config.MODEL.num_classes)
        if config.MODEL.backbone.frozen:
            unfreeze = False
            stage = str(config.MODEL.backbone.unfreeze_start_stage-1)
            for name, param in self.backbone.named_parameters():
                if stage in name:
                    unfreeze = True
                param.requires_grad=unfreeze
                if param.requires_grad and config.local_rank==0 and stage in name:
                    logger.info(f"{name} is set to be trainable.")

    def forward(self, x):
        x = self.backbone(x)
        x = self.head1(x)
        return x

class CLIP_ViT_cl(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.backbone = build_clip(config, logger)
        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()
        self.head = nn.Linear(config.MODEL.backbone.output_dim, config.MODEL.num_classes)
        # self.head2 = ArcMarginProduct(config.MODEL.backbone.output_dim, config.MODEL.num_classes)
        # self.head1 = ArcMarginProduct_subcenter(config.MODEL.backbone.output_dim, config.MODEL.num_classes, k=3)
        if config.MODEL.backbone.frozen:
            unfreeze = False
            stage = str(config.MODEL.backbone.unfreeze_start_stage-1)
            for name, param in self.backbone.named_parameters():
                if stage in name:
                    unfreeze = True
                param.requires_grad=unfreeze
                if param.requires_grad and config.local_rank==0 and stage in name:
                    logger.info(f"{name} is set to be trainable.")

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
        # x1 = self.head1(x)
        # x2 = self.head2(x)
        # return x1, x2

class ST_CLIP_ViT_cl(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.backbone = build_clip(config, logger)
        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()
        self.norm = nn.LayerNorm(config.MODEL.backbone.width)
        self.ST_Layers = Domain_Transformer(config.MODEL.ST.checkpoint, config.MODEL.backbone.width, config.MODEL.backbone.reduced_dim, config.MODEL.ST.centers, config.MODEL.ST.layers)
        self.neck = nn.Linear((config.MODEL.img_size//config.MODEL.backbone.patch_size)**2 + 1, config.MODEL.backbone.reduced_dim)
        self.head = nn.Linear(config.MODEL.backbone.width, config.MODEL.num_classes)
        if config.MODEL.backbone.frozen:
            unfreeze = False
            stage = str(config.MODEL.backbone.unfreeze_start_stage-1)
            for name, param in self.backbone.named_parameters():
                if stage in name:
                    unfreeze = True
                param.requires_grad=unfreeze
                if param.requires_grad and config.local_rank==0 and stage in name:
                    logger.info(f"{name} is set to be trainable.")

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.permute(0, 2, 1)
        x = self.neck(x)
        x = x.permute(0, 2, 1)
        x = self.ST_Layers(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x

class ST_Net_TIMM(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.backbone = timm.create_model(config.MODEL.backbone.model_name, pretrained=True)
        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()

        logger.info(f'Load {config.MODEL.backbone.model_name} from timm successfully !')
        # self.norm = nn.LayerNorm(config.MODEL.backbone.out_dims)
        # self.ST_Layers = Domain_Transformer(config.MODEL.ST.checkpoint, config.MODEL.backbone.out_dims, (config.MODEL.img_size//config.MODEL.backbone.patch_size**2 + 1, config.MODEL.ST.centers, config.MODEL.ST.layers)
        # self.neck = nn.Linear(config.MODEL.backbone.out_dims, config.MODEL.backbone.out_dims)
        self.head = nn.Linear(config.MODEL.backbone.out_dims, config.MODEL.len_embeddings)
        if config.MODEL.backbone.frozen:
            for param in self.backbone.parameters():
                param.requires_grad=False

    def forward(self, x):
        # x = self.backbone(x)
        x = self.backbone.forward_features(x)
        x = self.neck(x)
        x = self.ST_Layers(x)[:, 0, :]
        x = self.norm(x)
        x = self.head(x)
        return x

class VPT_Net(nn.Module):
    def __init__(self, config, logger):
        super().__init__()        
        self.backbone = Backbone(config, logger)
        self.g_pool = nn.AdaptiveAvgPool2d(1) if config.MODEL.feature_pool else nn.Identity()
        self.head = nn.Linear(config.MODEL.backbone.output_dim, config.MODEL.num_classes)
        if config.MODEL.backbone.frozen:
            self.backbone.net.p_tokens.requires_grad=True

    def forward_feature(self, x):
        x = self.backbone(x)
        feature = self.g_pool(x)
        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        logits = self.head(feature)
        return logits

def build_ST(config):
    ST_Module = Domain_Transformer(check_point = config.MODEL.ST.checkpoint, 
                                    width = config.MODEL.backbone.width, 
                                    len_token = config.MODEL.backbone.reduced_dim, 
                                    centers = config.MODEL.ST.centers, 
                                    dt_layers = config.MODEL.ST.layers,
                                    mlp_ratio = config.MODEL.ST.mlp_ratio)
    return ST_Module

def build_clip(config, logger):
    if 'ViT' in config.MODEL.backbone.model_name:
        model = VisionTransformer(
                num_stage=config.MODEL.backbone.num_stage,
                image_size=config.MODEL.img_size,
                patch_size=config.MODEL.backbone.patch_size,
                width=config.MODEL.backbone.width,
                layers=config.MODEL.backbone.layers,
                heads=config.MODEL.backbone.heads,
                mlp_ratio=config.MODEL.backbone.mlp_ratio,
                global_average_pool=config.MODEL.backbone.global_average_pool,
                attentional_pool=config.MODEL.backbone.attentional_pool,
                attn_pooler_heads=config.MODEL.backbone.attn_pooler_heads,
                n_queries=config.MODEL.backbone.n_queries,
                output_dim=config.MODEL.backbone.output_dim
              )
    elif 'eva_02' in config.MODEL.backbone.model_name:
        model = EVA_02(
            num_stage=config.MODEL.backbone.num_stage,
            checkpoint_start_stage=config.MODEL.backbone.checkpoint_start_stage-1,
            img_size=config.MODEL.img_size,
        )
    elif 'eva-decoup' in config.MODEL.backbone.model_name:
        model = EVAVisionTransformer_Decoup(
            num_stage=config.MODEL.backbone.num_stage,
            checkpoint_start_stage=config.MODEL.backbone.checkpoint_start_stage-1,
            shared_dpth=config.MODEL.backbone.shared_dpth, 
            indepdent_depth=config.MODEL.backbone.indepdent_depth, 
            num_decouplers=config.MODEL.backbone.num_decouplers,
            img_size=config.MODEL.img_size,
            patch_size=config.MODEL.backbone.patch_size,
            num_classes=config.MODEL.backbone.output_dim,
            use_mean_pooling=config.MODEL.backbone.use_mean_pooling,
            init_values=config.MODEL.backbone.init_values,
            patch_dropout=config.MODEL.backbone.patch_dropout,
            embed_dim=config.MODEL.backbone.width,
            depth=config.MODEL.backbone.depth,
            num_heads=config.MODEL.backbone.num_heads,
            mlp_ratio=config.MODEL.backbone.mlp_ratio,
            qkv_bias=config.MODEL.backbone.qkv_bias,
            drop_path_rate=config.MODEL.backbone.drop_path_rate,
            xattn=config.MODEL.backbone.xattn,
            rope=config.MODEL.backbone.rope,
            postnorm=config.MODEL.backbone.postnorm,
            pt_hw_seq_len=config.MODEL.backbone.pt_hw_seq_len,
            intp_freq= config.MODEL.backbone.intp_freq,
            naiveswiglu= config.MODEL.backbone.naiveswiglu,
            subln= config.MODEL.backbone.subln,
            norm_layer=partial(LayerNormFp32, eps=1e-6) if config.MODEL.type in ('fp16', 'bf16') else partial(LayerNorm, eps=1e-6)
        )
    elif 'eva-cl' in config.MODEL.backbone.model_name:
        model = EVAVisionTransformer(
            num_stage=config.MODEL.backbone.num_stage,
            checkpoint_start_stage=config.MODEL.backbone.checkpoint_start_stage-1,
            img_size=config.MODEL.img_size,
            patch_size=config.MODEL.backbone.patch_size,
            num_classes=config.MODEL.backbone.output_dim,
            use_mean_pooling=config.MODEL.backbone.use_mean_pooling,
            init_values=config.MODEL.backbone.init_values,
            patch_dropout=config.MODEL.backbone.patch_dropout,
            embed_dim=config.MODEL.backbone.width,
            depth=config.MODEL.backbone.depth,
            num_heads=config.MODEL.backbone.num_heads,
            mlp_ratio=config.MODEL.backbone.mlp_ratio,
            qkv_bias=config.MODEL.backbone.qkv_bias,
            drop_path_rate=config.MODEL.backbone.drop_path_rate,
            xattn=config.MODEL.backbone.xattn,
            rope=config.MODEL.backbone.rope,
            postnorm=config.MODEL.backbone.postnorm,
            pt_hw_seq_len=config.MODEL.backbone.pt_hw_seq_len,
            intp_freq= config.MODEL.backbone.intp_freq,
            naiveswiglu= config.MODEL.backbone.naiveswiglu,
            subln= config.MODEL.backbone.subln,
            norm_layer=partial(LayerNormFp32, eps=1e-6) if config.MODEL.type in ('fp16', 'bf16') else partial(LayerNorm, eps=1e-6)
        )
    elif 'convnext'in config.MODEL.backbone.model_name:
        model = TimmModel(
                model_name=config.MODEL.backbone.model_name,
                embed_dim=config.MODEL.backbone.embed_dim,
                image_size=config.MODEL.img_size,
                pool=config.MODEL.backbone.timm_pool,
                proj=config.MODEL.backbone.timm_proj,
                proj_bias=config.MODEL.backbone.timm_proj_bias,
                drop=config.MODEL.backbone.timm_drop,
                drop_path=config.MODEL.backbone.timm_drop_path,
                pretrained=config.MODEL.backbone.timm_model_pretrained,
                out_indices=config.MODEL.backbone.out_indices
        )
    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.backbone.model_name}")
    # model.load_state_dict(torch.load(config.MODEL.backbone.model_path, map_location='cpu'), strict=True)
    if logger is not None:
        logger.info(f"=> Load '{config.MODEL.backbone.model_path}' successfully")
    return model
