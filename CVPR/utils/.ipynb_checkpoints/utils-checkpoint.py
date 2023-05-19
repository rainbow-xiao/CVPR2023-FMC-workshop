import os
import torch
import random
import numpy as np
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from .loss import ArcFaceLoss, ArcFaceLossAdaptiveMargin, DenseCrossEntropy_Multi_Label, DenseCrossEntropy
from .AdamS import AdamS

def get_optim_from_config(config, model, mode):
    if mode == 'embed':
        param_dicts = [{'params':filter(lambda p:p.requires_grad, model.module.backbone.parameters()), 'lr':config.init_lr, 
                        'weight_stick_max':config.Optimizer.weight_stick_max, 'weight_stick_min':config.Optimizer.weight_stick_min, 
                        'weight_decay':config.Optimizer.weight_decay, 'stick_pow':config.Optimizer.stick_pow, 'enable_stick':True, 
                        'scale_coefficient':1},
                      {'params':filter(lambda p:p.requires_grad, model.module.head.parameters()), 'lr':config.init_lr, 
                       'weight_decay':config.Optimizer.weight_decay, 'enable_stick':False, 'scale_coefficient':1}]
        # param_dicts = [{'params':filter(lambda p:p.requires_grad, model.parameters()), 'lr':config.init_lr, 'weight_decay':config.Optimizer.weight_decay, 'scale_coefficient':1},]
    else:
        raise NotImplementedError(f"Unkown param_dicts")
    if config.Optimizer.name=='SGD':
        optimizer = optim.SGD(param_dicts)
    elif config.Optimizer.name=='Adam':
        optimizer = optim.Adam(param_dicts)
    elif config.Optimizer.name=='AdamW':
        optimizer = optim.AdamW(param_dicts)
    elif config.Optimizer.name=='AdamS':
        optimizer = AdamS(param_dicts)
    else:
        raise NotImplementedError(f"Unkown optimizer: {config.Optimizer.name}")
    return optimizer

def get_criterion_from_config(config):
    if config.Loss.name == 'ce_loss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif config.Loss.name == 'arcface_loss':
        criterion = ArcFaceLoss(s=config.Loss.s, m=config.Loss.m).cuda()
    elif config.Loss.name == 'one_hot_ce':
        criterion = DenseCrossEntropy().cuda()
    elif config.Loss.name == 'multi_ce':
        criterion = DenseCrossEntropy_Multi_Label().cuda()
    elif config.Loss.name == 'adaptive_arcface_loss':
        criterion = ArcFaceLossAdaptiveMargin(s=config.Loss.s, min_m=config.Loss.min_m, max_m=config.Loss.max_m).cuda()
    else:
        raise NotImplementedError(f"Unkown Loss: {config.Loss.name}")
    return criterion

def get_mul_criterion_from_config(config):
    if config.Loss1.name == 'ce_loss':
        criterion1 = nn.CrossEntropyLoss().cuda()
    elif config.Loss1.name == 'arcface_loss':
        criterion1 = ArcFaceLoss(s=config.Loss1.s, m=config.Loss1.m).cuda()
    elif config.Loss1.name == 'multi_ce':
        criterion1 = DenseCrossEntropy_Multi_Label().cuda()
    elif config.Loss1.name == 'adaptive_arcface_loss':
        criterion1 = ArcFaceLossAdaptiveMargin(s=config.Loss1.s, min_m=config.Loss1.min_m, max_m=config.Loss1.max_m).cuda()
    else:
        raise NotImplementedError(f"Unkown Loss: {config.Loss1.name}")
    
    if config.Loss2.name == 'ce_loss':
        criterion2 = nn.CrossEntropyLoss().cuda()
    elif config.Loss2.name == 'arcface_loss':
        criterion2 = ArcFaceLoss(s=config.Loss2.s, m=config.Loss2.m).cuda()
    elif config.Loss2.name == 'multi_ce':
        criterion2 = DenseCrossEntropy_Multi_Label().cuda()
    elif config.Loss2.name == 'adaptive_arcface_loss':
        criterion2 = ArcFaceLossAdaptiveMargin(s=config.Loss2.s, min_m=config.Loss2.min_m, max_m=config.Loss2.max_m).cuda()
    else:
        raise NotImplementedError(f"Unkown Loss: {config.Loss2.name}")
    return criterion1, criterion2

def config_from_name(name):
    if name == 'config_clip_vit':
        from config_clip_vit import get_config
        config = get_config()
    elif name == 'config_eva_vit':
        from config_eva_vit import get_config
        config = get_config()
    elif name == 'config_eva_vit_car_cl':
        from config_eva_vit_car_cl import get_config
        config = get_config()
    elif name == 'config_conv_car_cl':
        from config_clip_convnext_car_cl import get_config
        config = get_config()
    elif name == 'config_eva_02_car_cl':
        from config_eva_02_car_cl import get_config
        config = get_config()
    elif name == 'config_eva_vit_people_cl':
        from config_eva_vit_people_cl import get_config
        config = get_config()
    elif name == 'config_conv_people_cl':
        from config_clip_convnext_people_cl import get_config
        config = get_config()
    elif name == 'config_eva_02_people_cl':
        from config_eva_02_people_cl import get_config
        config = get_config()
    else:
        raise NotImplementedError(f"Unkown config_name: {name}")
    return config

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, save_path, optimizer=None, epoch=None, lr=None):
    if optimizer!=None:
        save_state = {'state_dict': model.module.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch,
                      'lr': lr}
    else:
        save_state = {'state_dict': model.module.state_dict(),
                      'optimizer': {},
                      'epoch': {},
                      'lr': {}}
    torch.save(save_state, save_path)

def load_ckpt_finetune(path, model, optimizer=None, logger=None, args=None):
    dicts = torch.load(path, map_location='cpu')
    model.load_state_dict(dicts['state_dict'], strict=True)
    if args.local_rank == 0:
        logger.info('Load pre-trained weight for finetuning successfully!')
        
    if optimizer != None:
        args.n_epochs = dicts['epoch']
        args.init_lr = dicts['lr']
        optimizer.load_state_dict(dicts['optimizer'], strict=True)
        if args.local_rank == 0:
            logger.info('Load dicts of optimizer for finetuning successfully!')

def get_train_epoch_lr(cur_epoch, steps, max_epoch, init_lr, iters_per_epoch=None):
    cur_step = (cur_epoch-1)*iters_per_epoch+steps
    total_step = max_epoch*iters_per_epoch
    return 0.5 * init_lr * (1.0 + np.cos(np.pi * cur_step / total_step))

def get_warm_up_lr(cur_epoch, steps, warm_up_epochs, init_lr, iters_per_epoch=None):
    cur_step = (cur_epoch-1)*iters_per_epoch+steps
    total_step = warm_up_epochs*iters_per_epoch
    alpha = cur_step / total_step
    factor = min(0.01 * (1.0 - alpha) + alpha, 1.)
    lr = init_lr*factor
    return lr

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr*pg['scale_coefficient']