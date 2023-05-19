import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable
from torch.utils.checkpoint import checkpoint

class Space_Attention(nn.Module):
    def __init__(self, in_dim, out_dim, centers):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = in_dim ** -0.5
        self.k_c = nn.Parameter(torch.randn(in_dim, centers))
        self.v_c = nn.Parameter(torch.randn(centers, in_dim, out_dim))

    def forward(self, x):
        attn = torch.mm(F.normalize(x, dim=-1), F.normalize(self.k_c, dim=0))
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.in_dim, self.out_dim)
        v = (attn*(self.v_c.expand_as(attn))).sum(1)
        x = x.unsqueeze(1)
        x = torch.bmm(x, v)
        return x[:, 0, :]

class Domain_Attention(nn.Module):
    def __init__(self, dim, len_token, centers, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.scale = dim ** -0.5
        self.len_token = len_token
        self.k_c = nn.Parameter(torch.randn(dim, centers))
        self.v_c = nn.Parameter(torch.randn(centers, len_token, len_token))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = x.mean(1)
        attn = torch.mm(F.normalize(q, dim=-1), F.normalize(self.k_c, dim=0))
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn.unsqueeze(-1).unsqueeze(-1).expand(B, -1, self.len_token, self.len_token)
        v = (attn*(self.v_c.unsqueeze(0).expand_as(attn))).sum(1)
        x = torch.bmm(v, x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DA_Resblock(nn.Module):
    def __init__(
            self,
            d_model: int,
            len_token: int,
            centers: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        self.attn = Domain_Attention(d_model, len_token, centers)

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Domain_Transformer(nn.Module):
    def __init__(self, check_point: list, width: int, len_token: int, centers: int, dt_layers: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.check_point = check_point
        self.da_resblocks = nn.ModuleList([DA_Resblock(width, len_token, centers, mlp_ratio, act_layer=act_layer) for i in range(dt_layers)])

    def forward(self, x: torch.Tensor):
        if self.check_point[0] and not torch.jit.is_scripting():
            for idx, dr in enumerate(self.da_resblocks):
                if idx<self.check_point[1]:
                    x = checkpoint(dr, x, preserve_rng_state=False)
                else:
                    x = dr(x)
        else:
            for dr in self.da_resblocks:
                x = dr(x)
        return x