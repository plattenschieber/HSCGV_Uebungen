#include "cudaTest.h"
#include <stdio.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct squareThrust
{
    __host__ __device__
        T operator()(const T& x) const {
            return x * x;
        }
};

// cuda kernel that performs an operation
// (all arguments need to be allocated in advance)
__global__ void square (float * d_out, float * d_in) {
  int id = threadIdx.x;
  float f = d_in[id];
  d_out[id] = f*f;
}

int launchKernel () {
    thrustSortTest();
    return 0;
}

void cudaDevicesTest() {
    // test how many devices we have
    int devices;
    cudaGetDeviceCount(&devices);
    printf("We have %i device(s) on this machine\n", devices);
}

void cudaSquareTest() {
    // host variable for input
    float h_in[20];
    for (int i=0; i<20; i++)
    {
      h_in[i] = float(i);
    }
    float h_out[20];

    // we need in- and output variables on the device, too
    float *d_in;
    float *d_out;

    // get some space on the GPU
    cudaMalloc((void**) &d_in, 20*sizeof(float));
    cudaMalloc((void**) &d_out, 20*sizeof(float));

    // copy device data to host with according copy type
    cudaMemcpy(d_in, h_in, 20*sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel
    square<<<1, 20>>>(d_out, d_in);

    // copy back data to the CPU
    cudaMemcpy(h_out, d_out, 20*sizeof(float), cudaMemcpyDeviceToHost);

    // print data
    for (int i=0; i<20; i++)
    {
      printf("%f\n", h_out[i]);
    }

    // free data on GPU
    cudaFree(d_in);
    cudaFree(d_out);
}

void thrustVectorTest() {
    // now do the same with thrust
    // initialize host array
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

    // setup arguments
    squareThrust<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );

    std::cout << "norm: " <<  norm << std::endl;
}

void thrustSortTest() {
    // generate 128M random numbers serially
    clock_t start = clock();
    thrust::host_vector<int> h_vec(32 << 22);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // sort data on the device (846M keys per second on GeForce GTX 480)
    std::sort(h_vec.begin(), h_vec.end());
    clock_t end = clock();

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());
    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    clock_t end2 = clock();

    float seqSec = (float)(end - start) / CLOCKS_PER_SEC;
    float parSec = (float)(end2 - end) / CLOCKS_PER_SEC;
    std::cout << "it took " << parSec << " for the parallel algorithm" << std::endl;
    std::cout << "it took " << seqSec << " for the sequential algorithm" << std::endl;
}
