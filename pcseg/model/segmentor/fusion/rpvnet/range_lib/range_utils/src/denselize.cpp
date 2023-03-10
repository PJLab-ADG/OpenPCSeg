#include <torch/torch.h>
#include <vector>
#include "denselize_gpu.h"

at::Tensor denselize_forward(
    const at::Tensor feat,
    const at::Tensor count_map,
    const at::Tensor pxpy
)
{
    int B = count_map.size(0);
    int W = count_map.size(2);
    int H = count_map.size(1);
    int C = feat.size(1);
    int N = pxpy.size(0);
    at::Tensor feature_map = torch::zeros({B,C,H,W}, at::device(pxpy.device()).dtype(at::ScalarType::Float));
    denselize_wrapper(N,C,H,W,feat.data_ptr<float>(),count_map.data_ptr<int>(),pxpy.data_ptr<int>(),feature_map.data_ptr<float>());
    return feature_map;    
}

at::Tensor denselize_backward(
    const at::Tensor top_grad,
    const at::Tensor count_map,
    const at::Tensor pxpy
)
{
    int B = count_map.size(0);
    int W = count_map.size(2);
    int H = count_map.size(1);
    int C = top_grad.size(1);
    int N = pxpy.size(0);
    at::Tensor bottom_grad = torch::zeros({N, C}, at::device(pxpy.device()).dtype(at::ScalarType::Float));
    denselize_grad_wrapper(N,C,H,W,top_grad.data_ptr<float>(),count_map.data_ptr<int>(),pxpy.data_ptr<int>(),bottom_grad.data_ptr<float>());
    return bottom_grad;
}