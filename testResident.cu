#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxResidentThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    printf("Max thread residenti simultaneamente: %d\n", maxResidentThreads);
    return 0;
}
