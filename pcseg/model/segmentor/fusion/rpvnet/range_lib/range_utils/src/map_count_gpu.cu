#include <stdio.h>
#include <stdlib.h>
#include <cmath>

__global__ void map_count_kernel(int N, int H, int W,const int *__restrict__ data, int *__restrict__ out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        int bs = data[3*i];
        int px = data[3*i + 1];
        int py = data[3*i + 2];
        // printf("bs: %i px: %i py: %i h: %i w: %i\n",bs,px,py,H,W);
        if(px >= 0 && py >= 0)atomicAdd(&out[bs*(H*W) + py*W + px], 1);
    }
}

void map_count_wrapper(int N, int H,int W,const int * data, int * out){
    map_count_kernel<<<ceil((double)N/512), 512>>>(N, H, W, data, out);
}