#include "lbmCu.h"
// cuda kernel that performs an operation
// (all arguments need to be allocated in advance)
__global__ void square (float * d_out, float * d_in) {
  int id = threadIdx.x;
  float f = d_in[id];
  d_out[id] = f*f;
}

int launchKernel () {
    cudaSquareTest();
    return 0;
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

