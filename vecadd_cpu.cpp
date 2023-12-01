//
// Created by Elliot Warren on 12/1/2023.
// Bulk of the code copied from https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// edits made include addition of timer, calculations for time, MFLOPs, memory bandwidth
// also tweaked N value.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

// function to add the elements of two arrays
void add(int n, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main()
{
    //from https://www.amd.com/en/products/cpu/amd-epyc-7763
    double capacity = 204.8e9; // Peak memory bandwidth in bytes/sec.

    int N = 1<<29; // 512M elements

    auto *x = new float[N];
    auto *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on 512M elements on the CPU
    add(N, x, y);

    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();

    double mflops = (N/1e6) / duration;

    double bytes = N * sizeof(uint64_t);

    double memoryBandwidthUtilized = ((((bytes / 1e9) / duration) / capacity) * 100) * 1e9; // % of memory bandwidth utilized

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = std::fmax(maxError, std::fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::setprecision(5) << std::endl;

    printf("Time elapsed: %f seconds\n", duration);
    printf("MFLOP/s: %f\n", mflops);
    printf("% Memory bandwidth utilized: %f\n", memoryBandwidthUtilized);

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}