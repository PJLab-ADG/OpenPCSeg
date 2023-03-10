import range_utils.nn.functional as rnf
import torch
from torch.autograd.gradcheck import gradcheck
from torch.autograd import Variable



if __name__ == "__main__":
    print('Test map_count ..............')
    pxpy = torch.Tensor([[0,2,2],[0,2,2],[1,1,0]]).int().cuda()
    max_bs = 2
    w = 4
    h = 5
    count_map = rnf.map_count(pxpy, max_bs, h, w)
    print(count_map)
    
    print('Test denselize ..............')
    pf = torch.Tensor([[1,1,2],[1,2,2],[4,4,4]]).float().cuda()
    pf.requires_grad = True
    # print(feature_map.size)
    inputs = (pf,count_map,pxpy)
    f_m = rnf.denselize(*inputs)
    test = gradcheck(rnf.denselize,inputs,eps=1e-3,atol=1e-4)
    print(test)


