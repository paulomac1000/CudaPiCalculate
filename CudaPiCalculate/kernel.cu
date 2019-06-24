#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>

using namespace thrust;
using namespace std;

#define W 1000 //liczba w¹tkow
#define K 100 //liczba generowanych punktów przez kazdy w¹tek

#pragma region handle randoms

__global__ void __launch_bounds__(1024, 2) initRand(unsigned long long seed, curandState_t *state, int n)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < n) {
		curand_init(seed, tid, 0, &state[tid]);
	}
}

__global__ void computeRandom(curandState_t *state, double *tab, int n)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid<n) {
		curandState_t st = state[tid];

		//przerabiam int na double
		tab[tid] = (double)curand(&st) / 10000.0;
		tab[tid] -= (int)tab[tid];

		state[tid] = st;
	}
}

device_vector<double> generateRands()
{
	unsigned long int count = W * K * 2; //2*K liczb dla W w¹tków
	unsigned int bs = 1024; //Rozmiar bloku
	unsigned long long int seed = 12346;
	device_vector<curandState_t> d_states; //stany generatora  liczb prseudolosowych dla ka¿dego w¹tku
	device_vector<double> randoms; //wynikowe liczby

	d_states.resize(count);
	randoms.resize(count);

	dim3 grid = dim3(ceil((double)count / (double)bs));

	initRand << <grid, bs >> > (seed, d_states.data().get(), d_states.size());
	computeRandom << <grid, bs >> > (d_states.data().get(), randoms.data().get(), randoms.size());
	cudaDeviceSynchronize();

	return randoms;
}

#pragma endregion

__global__ void kernelCount(int *a, double *randoms, int w, int k)
{
	int threadId = threadIdx.x;

	int s = 0;

	if (threadId < w)
	{
		for (int i = 0; i < k; i++)
		{
			double x = randoms[threadId * i * 2];
			double y = randoms[threadId * i * 2 + 1];

			if ((x * x) + (y * y) < 1)
				s++;
		}

		a[threadId] = s;
	}
}

unsigned long sumArray(int *a, int num_elements)
{
	int i = 0;
	unsigned long sum = 0;
	for (i = 0; i<num_elements; i++)
		sum = sum + a[i];

	return(sum);
}

int main()
{
	auto rands = generateRands();
	cout << "Liczby losowe wygenerowano pomyslnie" << endl;

	/*for (int i = 0; i < rands.size(); i++) {
		cout << "rands[" << i << "] = " << rands[i] << endl;
	}*/

	int *A;
	cudaMallocManaged(&A, W * sizeof(int));

	kernelCount <<< 1, W >>> (A, rands.data().get(), W, K);

	cudaDeviceSynchronize();

	/*for (int i = 0; i < W; i++) {
		cout << "a[" << i << "] = " << A[i] << endl;
	}*/

	unsigned long sum = sumArray(A, W);
	double pi = ((double)sum * 4.0) / (double)(W * K);

	cout << "PI = " << pi << endl;

	cudaFree(A);

	return 0;
}