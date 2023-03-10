#include <torch/torch.h>
#include <vector>
#include "map_count_gpu.h"

at::Tensor map_count_forward(
    const at::Tensor pxpy, // N * 3 : batch_id,px,py
    const int max_bs,
    const int h, // count map height
    const int w // count map width
)
{
    int N = pxpy.size(0);
    at::Tensor count_map = torch::zeros({max_bs,h,w}, at::device(pxpy.device()).dtype(at::ScalarType::Int));
    // pxpy.print();
    map_count_wrapper(N, h, w, pxpy.data_ptr<int>(), count_map.data_ptr<int>());
    return count_map;
}