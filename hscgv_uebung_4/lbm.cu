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
