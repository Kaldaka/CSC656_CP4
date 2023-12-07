//
// Created by Elliot Warren on 12/1/2023.
// Bulk of the code copied from https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// edits made include addition of timer, calculations for time, MFLOPs, memory bandwidth
// also tweaked N value.
//

#include <cmath>
#include <iostream>
#include <iomanip>

// function to add the elements of two arrays
//global specifier to add code to gpu kernel
__global__
void add(int n, const float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main()
{
    //from https://www.amd.com/en/products/cpu/amd-epyc-7763
    //double capacity = 204.8e9; // Peak memory bandwidth in bytes/sec.

    int deviceID = 0;
    int N = 1<<29; // 512M elements
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout << "number of blocks: " << numBlocks << std::endl;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemPrefetchAsync((void *)x, N*sizeof(float), deviceID);
    cudaMemPrefetchAsync((void *)y, N*sizeof(float), deviceID);

    // Run kernel on 512M elements on the GPU
    add<<<numBlocks, blockSize>>>(N, x, y);

    /*double duration = std::chrono::duration<double>(end - start).count();

    double mflops = (N/1e6) / duration;

    double bytes = N * sizeof(uint64_t);

    double memoryBandwidthUtilized = ((((bytes / 1e9) / duration) / capacity) * 100) * 1e9; // % of memory bandwidth utilized

    printf("Time elapsed: %f seconds\n", duration);
    printf("MFLOP/s: %f\n", mflops);
    printf("% Memory bandwidth utilized: %f\n", memoryBandwidthUtilized);*/

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = std::fmax(maxError, std::fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::setprecision(5) << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}