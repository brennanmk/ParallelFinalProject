
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t addWithCuda(int N, int a, int b, long double *total);


__global__ void addKernel(int N, int a, int b, long double *total)
{
    long double temp = 0;
    int i = threadIdx.x;

    for (i = 0; i < N; i++) //perform provided calculations N times (add to total each time)
    {
        int numGen = (rand() % (b - a + 1)) + a;

        long double powerOf = (-1 * powl(numGen, 2) / 2);
        temp += ((1 / sqrtl(2 * 3.14)) * ((powl(2.718, powerOf))));
    }

    *total += temp;
}

int main(int argc, char* argv[])
{
    long double *total; //variable to store total
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int N = atoi(argv[3]);

    cudaMallocManaged(&total, sizeof(long double));

    // Add vectors in parallel.
    addKernel << <1, 256 >> > (N, a, b, total);


    printf("%lf", total);

    return 0;
}
