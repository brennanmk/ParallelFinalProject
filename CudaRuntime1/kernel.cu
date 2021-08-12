
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void monteCarlo(long timeVal, int N, int a, int b, long double* answer)
{
	long double temp = 0;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	long double F; //variable for final integral
	int count = 0;



	curandState_t state;
	curand_init((timeVal * index), /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	if (i < N) {
		int numGen = (curand(&state) % (b - a + 1)) + a;
		count++;
		double powerOf = (-1 * pow(numGen, 2) / 2);

		answer[i] = ((1 / sqrt(2 * 3.14)) * ((pow(2.718, powerOf))));

	}

}

int main() {
	int a = -5;
	int b = 5;
	int N = 100000000;
	long timeVal = time(NULL);
	double timeAvg;

	int size = N * sizeof(long double);

	long double* total = 0;
	long double* d_total;
	long double F = 0;
	long double temp = 0;


	clock_t startTime = clock(); //record start time (function found from https://en.cppreference.com/w/c/chrono/clock_t)

	total = (long double*)malloc(size);
	cudaMalloc((void**)&d_total, size);

	int nblocks = (N + 511) / 512;


	cudaMemcpy(d_total, total, size, cudaMemcpyHostToDevice);

	monteCarlo <<<nblocks, 512 >>> (timeVal, N, a, b, d_total);

	cudaMemcpy(total, d_total, size, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N; i++) {
		temp += total[i];

	}
	F = (((long double)b - (long double)a) / (long double)N) * temp;


	clock_t endTime = clock(); //end time (function found from https://en.cppreference.com/w/c/chrono/clock_t)
	timeAvg = ((double)(endTime - startTime)) / CLOCKS_PER_SEC; //add the elasped time to timeAvg


	printf("%lf\n", F);
	printf("Run Time = %fs\n", timeAvg); //Print average time

	cudaFree(d_total);

	return 0;
}
