#ifndef _DENSE_LIZE
#define _DENSE_LIZE
#include <torch/torch.h>
#include <vector>

void denselize_wrapper(int N, int C, int H, int W, const float * feat, const int * counts ,const int * pxpy, float * out);
void denselize_grad_wrapper(int N, int C, int H, int W, const float * top_grad, const int * count_map, const int * pxpy, float * bottom_grad);

at::Tensor denselize_forward(
    const at::Tensor feat,
    const at::Tensor count_map,
    const at::Tensor pxpy
);
at::Tensor denselize_backward(
    const at::Tensor top_grad,
    const at::Tensor count_map,
    const at::Tensor pxpy
);

#endif