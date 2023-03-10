# raw point
# ...

# range view
from .range.rangenet.model.semantic.rangenet import RangeNet
from .range.salsanext.model.semantic.salsanext import SalsaNext
from .range.fidnet.model.semantic.fidnet import FIDNet
from .range.cenet.model.semantic.cenet import CENet

# bird's eye view
# ...

# voxel
from .voxel.cylinder3d import Cylinder_TS
from .voxel.cylinder3d.cylinder_ts import Cylinder_TS
from .voxel.minkunet.minkunet import MinkUNet

# multi-view fusion
from .fusion.spvcnn.spvcnn import SPVCNN #, MinkUNet
from .fusion.rpvnet.rpvnet import RPVNet



__all__ = {
    # raw point
    # ...

    # range view
    'RangeNet++': RangeNet,
    'SalsaNext': SalsaNext,
    'FIDNet': FIDNet,
    'CENet': CENet,

    # bird's eye view
    # ...

    # voxel
    'Cylinder_TS': Cylinder_TS,
    'MinkUNet': MinkUNet,

    # multi-view fusion
    'SPVCNN': SPVCNN,
    'RPVNet': RPVNet,
}


def build_segmentor(model_cfgs, num_class):
    model = eval(model_cfgs.NAME)(
        model_cfgs=model_cfgs,
        num_class=num_class,
    )

    return model
