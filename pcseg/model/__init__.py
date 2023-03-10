from collections import namedtuple

import numpy as np
import torch

from .segmentor import build_segmentor
from torchsparse import SparseTensor


def build_network(model_cfgs, num_class):
    model = build_segmentor(
        model_cfgs=model_cfgs,
        num_class=num_class,
    )
    return model

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda()
        elif isinstance(val, SparseTensor):
            batch_dict[key] = val.cuda()
        elif isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).cuda()
        elif isinstance(val, dict):
            for k, v in val.items():
                batch_dict[key][k] = v.cuda()
        elif isinstance(val, list):
            batch_dict[key] = val
        else:
            raise ValueError("Invalid type of batch_dict")

def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
