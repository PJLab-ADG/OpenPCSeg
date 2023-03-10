from torch.autograd import Function

import rangelib_cuda

__all__ = ['denselize']

class DenselizeGPU(Function):
    @staticmethod
    def forward(ctx, feat, count_map, pxpy):
        '''
        args:
            feat: torch.Float32 tensor N X C 
            count_map: int max batch size
            pxpy: torch.Int32 tensor N X 3 (mean batch_id, px, py)
                px range:0~w 
                py range:0~h
        return:
            outs:torch.Int32 tensor B C X H X W
        '''
        outs = rangelib_cuda.denselize_forward(feat, count_map, pxpy)
        ctx.for_backwards = (count_map.int().contiguous(), pxpy.int().contiguous())
        return outs

    @staticmethod
    def backward(ctx, top_grad):
        count_map, pxpy = ctx.for_backwards
        bottom_grad = rangelib_cuda.denselize_backward(top_grad.float().contiguous(),count_map,pxpy)
        return bottom_grad, None, None


denselize_gpu = DenselizeGPU.apply

def denselize(feat, count_map, pxpy):
    return denselize_gpu(feat, count_map, pxpy)
