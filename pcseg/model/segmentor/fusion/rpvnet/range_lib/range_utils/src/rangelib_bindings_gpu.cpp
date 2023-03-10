#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "map_count_gpu.h"
#include "denselize_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("map_count_forward", &map_count_forward, "Map counting forward (CUDA)");
    m.def("denselize_forward", &denselize_forward, "Denselize forward (CUDA)");
    m.def("denselize_backward", &denselize_backward, "Denselize backward (CUDA)");
    
}



