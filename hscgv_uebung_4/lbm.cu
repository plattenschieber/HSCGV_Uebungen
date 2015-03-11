#include "lbm.h"
#include <cuda.h>

// constants
//! 0 = no periodic boundaries, 1 = periodic boundaries -- non-periodic is faster
static const int eriodicBoundaries = 0;
//! dimensionaly of domain, has to be 3
static const int D = 3;
//! size of neighbourhood of a cell, has to be 19
static const int Q = 19;

// we can save constants on the GPU in an extra space with a lot faster access
__constant__ float d_w[Q];
__constant__ int d_e[Q][D];
__constant__ int d_invDir[Q];

//! we need some kind of initialization of our device
void LBMD3Q19::intializeCuda() {
    // get some space for our arrays
    cudaMalloc((void **) &d_flags, sizeof(char) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_velocity, sizeof(float3) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_u, sizeof(float3) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_density, sizeof(float) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_cells[0], sizeof(float) * m_width * m_height * m_depth * Q);
    cudaMalloc((void **) &d_cells[1], sizeof(float) * m_width * m_height * m_depth * Q);

    // use cpyToSymbol for known sizes (LEGACY CODE - WORKS ONLY WITH CUDA <= 5.5)
    cudaMemcpyToSymbol(d_w, w.w, sizeof(float)*Q);
    cudaMemcpyToSymbol(d_e, e, sizeof(int3)*Q);
    cudaMemcpyToSymbol(d_invDir, invDir, sizeof(int)*Q);

    // copy data from host to device
    cudaMemcpy(d_flags, m_flags, m_width * m_height * m_depth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity, m_velocity, sizeof(float3) * m_width * m_height * m_depth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells[0], m_cells[0], sizeof(float) * m_width * m_height * m_depth * Q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells[1], m_cells[1], sizeof(float) * m_width * m_height * m_depth * Q, cudaMemcpyHostToDevice);
}

//! collide implementation with CUDA
void LBMD3Q19::collideCuda() {

}

//! streaming with CUDA
void LBMD3Q19::streamCuda() {

}

//! compute densities and velocities with CUDA
void LBMD3Q19::analyzeCuda() {

}

//! compute minimum and maximum density and velocity with CUDA
void LBMD3Q19::minMaxCpu() {

}
