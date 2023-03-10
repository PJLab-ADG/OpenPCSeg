#include <stdio.h>
#include <stdlib.h>
#include <cmath>

__global__ void denselize_kernel(int N,int C,int H, int W,const float *__restrict__ feat, const int *__restrict__ count_map ,const int * __restrict__ pxpy,float * __restrict__ out){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / C; //对应点索引
    int j = index % C; //对应点的特征索引
    if(i < N){
        int bs = pxpy[3*i]; //i点的batch id
        int px = pxpy[3*i + 1]; //i点在feature map上的x坐标
        int py = pxpy[3*i + 2]; //i在feature map上的y坐标
        int coord_pos = py*W + px;
        int pos = bs*H*W + coord_pos; // i点在pxpy对应的索引
        int out_pos = bs*C*H*W + j*H*W + coord_pos;
        if(pos < 0 || count_map[pos] == 0)return; //边界判断 这里没有判断上界 TODO
        atomicAdd(&out[out_pos], feat[i*C+j] / float(count_map[pos]));
    }
}

__global__ void denselize_grad_kernel(int N, int C, int H, int W, const float *__restrict__ top_grad, const int *__restrict__ count_map, const int *__restrict__ pxpy, float *__restrict__ bottom_grad){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / C;
    int j = index % C;
    if(i < N){
        int bs = pxpy[3*i]; //i点的batch id
        int px = pxpy[3*i + 1]; //i点在feature map上的x坐标
        int py = pxpy[3*i + 2]; //i在feature map上的y坐标
        int coord_pos = py*W + px;
        int pos = bs*W*H + coord_pos; // i点在pxpy对应的索引
        int out_pos = bs*C*H*W + j*H*W + coord_pos;
        atomicAdd(&bottom_grad[i*C + j], top_grad[out_pos] / float(count_map[pos]));
    }
}

void denselize_wrapper(int N,int C,int H, int W, const float * feat, const int * count_map ,const int * pxpy,float * out){
    denselize_kernel<<<N,C>>>(N,C,H,W,feat,count_map,pxpy,out);
}

void denselize_grad_wrapper(int N, int C, int H, int W, const float * top_grad, const int * count_map, const int * pxpy, float * bottom_grad){
    denselize_grad_kernel<<<N,C>>>(N, C, H , W, top_grad, count_map, pxpy, bottom_grad);
}