import math

import torch
from torch.optim import Optimizer


class AdamS(Optimizer):
    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, enable_stick=True,
            weight_stick_max=1, weight_stick_min=1e-2, weight_decay=0.01, stick_pow=1.0):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_stick_max=weight_stick_max, weight_stick_min=weight_stick_min, 
            weight_decay=weight_decay, enable_stick=enable_stick, stick_pow=stick_pow)
        super().__init__(params, defaults)