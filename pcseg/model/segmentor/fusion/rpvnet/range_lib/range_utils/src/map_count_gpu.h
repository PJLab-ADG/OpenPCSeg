#ifndef _SPARSE_MAP_COUNT
#define _SPARSE_MAP_COUNT
#include <torch/torch.h>
#include <vector>

//CUDA forward declarations
void map_count_wrapper(int N, int H, int W, const int * data, int * out);
at::Tensor map_count_forward(
    const at::Tensor pxpy, 
    const int max_bs,
    const int h,
    const int w
);
#endif