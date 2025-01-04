// test.cu
#include <stdio.h>

__global__ void dummyKernel() {
    // nada
}

int main() {
    dummyKernel<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("Hola desde CUDA\n");
    return 0;
}
