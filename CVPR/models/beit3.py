import math
import torch
import torch.nn as nn
from .torchscale.model.BEiT3 import BEiT3
from .torchscale.architecture.config import EncoderConfig

class Beit3(nn.Module):
    def __init__(self,**kargs):
        super().__init__()
        args = EncoderConfig(**kargs)
        self.beit3 = BEiT3(args)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        return x
