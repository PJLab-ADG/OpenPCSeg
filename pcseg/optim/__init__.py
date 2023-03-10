from functools import partial

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    
    if optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optim_cfg.LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM,
        )
    
    elif optim_cfg.OPTIMIZER == 'sgd_fc':
        base_dict = []
        for name, p in model.named_parameters():
            if "classifier" not in name:
                base_dict.append({'params': p})
        base_dict.append({'params': model.classifier.parameters(), 'lr': optim_cfg.LR * 10})
        optimizer = optim.SGD(
            base_dict,
            lr=optim_cfg.LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM,
        )

    elif optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optim_cfg.LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
        )
    
    elif optim_cfg.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optim_cfg.LR,
            betas=(optim_cfg.BETA1, optim_cfg.BETA2),
            weight_decay=optim_cfg.WEIGHT_DECAY,
            eps=optim_cfg.EPS,
        )
    
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    
    else:
        raise NotImplementedError
    
    return optimizer


def linear_warmup_with_cosdecay(cur_step, warmup_steps, total_steps, min_scale=1e-5):
    if cur_step < warmup_steps:
        return (1 - min_scale) * cur_step / warmup_steps + min_scale
    else:
        ratio = (cur_step - warmup_steps) / total_steps
        return (1 - min_scale) * 0.5 * (1 + np.cos(np.pi * ratio)) + min_scale


def cos_warmup_with_cosdecay(cur_step, warmup_steps, total_steps, min_scale=1e-5):
    if cur_step < warmup_steps:
        return (1 - min_scale) * (1 - np.cos(np.pi * cur_step / warmup_steps)) / 2 + min_scale
    else:
        ratio = (cur_step - warmup_steps) / total_steps
        return (1 - min_scale) * 0.5 * (1 + np.cos(np.pi * ratio)) + min_scale


def linear_warmup_with_stepdecay(cur_step, warmup_steps, total_steps, decay_steps, decay_scales):
    if cur_step < warmup_steps:
        return cur_step / warmup_steps
    else:
        cur_decay = 1
        for i in range(len(decay_steps)):
            if cur_step >= decay_steps[i]:
                cur_decay = cur_decay * decay_scales[i]
        return cur_decay


def coswarmup_with_stepdecay(cur_step, warmup_steps, total_steps, decay_steps, decay_scales):
    if cur_step < warmup_steps:
        return (1 - np.cos(np.pi * cur_step / warmup_steps)) / 2
    else:
        cur_decay = 1
        for i in range(len(decay_steps)):
            if cur_step >= decay_steps[i]:
                cur_decay = cur_decay * decay_scales[i]
        return cur_decay


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, optim_cfg) -> torch.optim.lr_scheduler.LambdaLR:

    total_steps = total_iters_each_epoch * total_epochs

    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    
    else:
        warmup_steps = optim_cfg.WARMUP_EPOCH * total_iters_each_epoch
        total_steps = total_epochs * total_iters_each_epoch

        if optim_cfg.SCHEDULER == 'linear_warmup_with_cosdecay':
            lr_scheduler = lr_sched.LambdaLR(
                optimizer,
                lr_lambda=lambda x: linear_warmup_with_cosdecay(x, warmup_steps, total_steps),
            )
        
        elif optim_cfg.SCHEDULER == 'cos_warmup_with_cosdecay':
            lr_scheduler = lr_sched.LambdaLR(
                optimizer,
                lr_lambda=lambda x: cos_warmup_with_cosdecay(x, warmup_steps, total_steps),
            )
        
        elif optim_cfg.SCHEDULER == 'linear_warmup_with_stepdecay':
            decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_EPOCHS]
            assert len(optim_cfg.DECAY_SCALES) == len(optim_cfg.DECAY_EPOCHS), "DECAY_SCALES not match the DECAY_EPOCHS"
            lr_scheduler = lr_sched.LambdaLR(
                optimizer,
                lr_lambda=lambda x: linear_warmup_with_stepdecay(
                    x, warmup_steps, total_steps, decay_steps, optim_cfg.DECAY_SCALES),
                )
        
        elif optim_cfg.SCHEDULER == 'coswarmup_with_stepdecay':
            decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_EPOCHS]
            assert len(optim_cfg.DECAY_SCALES) == len(optim_cfg.DECAY_EPOCHS), "DECAY_SCALES not match the DECAY_EPOCHS"
            lr_scheduler = lr_sched.LambdaLR(
                optimizer,
                lr_lambda=lambda x: coswarmup_with_stepdecay(
                    x, warmup_steps, total_steps, decay_steps, optim_cfg.DECAY_SCALES),
                )
        
        elif optim_cfg.SCHEDULER == 'onecycle':
            lr_scheduler = lr_sched.OneCycleLR(
                optimizer,
                max_lr=optim_cfg.LEARNING_RATE,
                epochs=total_epochs,
                steps_per_epoch=total_iters_each_epoch,
                pct_start=0.2,
                anneal_strategy='cos',
                cycle_momentum=True,
                div_factor=25.0,
                final_div_factor=100.0,
            )

        else:
            raise NotImplementedError("Not Supported SCHEDULER")
    
    return lr_scheduler
