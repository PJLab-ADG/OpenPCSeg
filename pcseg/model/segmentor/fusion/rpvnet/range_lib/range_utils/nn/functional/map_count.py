from torch.autograd import Function

import rangelib_cuda

__all__ = ['map_count']

class MapCountGPU(Function):
    @staticmethod
    def forward(ctx, pxpy, max_bs, h, w):
        '''
        args:
            pxpy: torch.Int32 tensor N X 3 (mean batch_id, px, py)
                px range:0~w 
                py range:0~h
            max_bs: int max batch size
            h：count map height
            w: count map width
        return:
            outs:torch.Int32 tensor B X H X W
                the counts of the pxpy in count map
        '''
        outs = rangelib_cuda.map_count_forward(pxpy.contiguous(), max_bs, h, w)
        return outs

map_count_gpu = MapCountGPU.apply

def map_count(pxpy, max_bs, h, w):
    return map_count_gpu(pxpy, max_bs, h, w)
