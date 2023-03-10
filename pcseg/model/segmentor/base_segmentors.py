import torch
import torch.nn as nn
import os


class BaseSegmentor(nn.Module):
    def __init__(
        self,
        model_cfgs,
        num_class: int,
    ):
        super().__init__()
        self.model_cfgs = model_cfgs
        self.num_class = num_class

    def load_params(self, model_state_disk, strict=False):
        my_model_dict = self.state_dict()
        part_load = {}
        for k in model_state_disk.keys():
            value = model_state_disk[k]
            if k.startswith("module."):
                k = k[len("module."):]
            if k in my_model_dict and my_model_dict[k].shape == value.shape:
                part_load[k] = value

        return self.load_state_dict(part_load, strict=strict)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        model_state_disk = torch.load(filename, map_location=loc_type)
        if 'model_state' in model_state_disk:
            model_state_disk = model_state_disk['model_state']
        msg = self.load_params(model_state_disk)
        logger.info(f"==> Done {msg}")

    def forward(self, batch_dict):
        raise NotImplementedError
