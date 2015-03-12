#include "lbm.h"
#include <cuda.h>

// constants
//! 0 = no periodic boundaries, 1 = periodic boundaries -- non-periodic is faster
static const int PeriodicBoundaries = 0;
//! dimensionaly of domain, has to be 3
static const int D = 3;
//! size of neighbourhood of a cell, has to be 19
static const int Q = 19;
//! cell types
enum CellFlags {
    //! a wet cell
    CellFluid = 0,
    //! a wall cell, flow bounces back
    CellNoSlip,
    //! fixed velocity cell
    CellVelocity
};


// we can save constants on the GPU in an extra space with a lot faster access
__constant__ float d_w[Q];
__constant__ int d_e[Q][D];
__constant__ int d_invDir[Q];
__constant__ float d_omega;

size_t __device__ index(int i, int j, int k) {

    if(PeriodicBoundaries) {
        i = (i+gridDim.x)%gridDim.x;
        j = (j+blockDim.x)%blockDim.x;
        k = (k+blockDim.y)%blockDim.y;
    }
    return i + blockDim.x*(j + blockDim.y*size_t(k));
}

size_t __device__ index(int i, int j, int k, int l) {
    if(PeriodicBoundaries) {
        i = (i+gridDim.x)%gridDim.x;
        j = (j+blockDim.x)%blockDim.x;
        k = (k+blockDim.y)%blockDim.y;
    }
#ifdef INNER_INDEX_DISTRIBUTION
    return l + Q*(i + blockDim.x*(j + blockDim.y*size_t(k)));
#else
    return i + blockDim.x*(j + blockDim.y*(size_t(k) + gridDim.x*l));
#endif
}

__global__ void collideCuda(float *d_cellsCur, char *d_flags, float3 *d_velocity) {
    // get the current thread position
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;

    // in case we have no periodic boundaries, the threads on the edges don't have anything to do
    if (!PeriodicBoundaries) {
        if (i==0 || i==blockDim.x-1 || j==0 || j==blockDim.y-1 || k==0 || k==gridDim.x-1)
            return;
    }

    // nothing to do for NoSlip cells
    const int flag = d_flags[index(i,j,k)];
    if (flag == CellNoSlip)
        return;

    // compute density and velocity in cell
    float density = 0.f;
    float3 u = make_float3(0.f, 0.f, 0.f);
    for(int l=0; l<Q; ++l) {
        const float weight = d_cellsCur[index(i,j,k,l)];
        density += weight;
        u.x += d_e[l][0] * weight;
        u.y += d_e[l][1] * weight;
        u.z += d_e[l][2] * weight;
    }

    // override velocity for Velocity cells
    if (flag == CellVelocity) {
        u = d_velocity[index(i,j,k)];
    }

    // collision
    for(int l=0; l<Q; ++l) {
        float dot = 0.f;
        float uu = 0.f;
        dot += d_e[l][0] * u.x;
        uu += u.x * u.x;
        dot += d_e[l][1] * u.y;
        uu += u.y * u.y;
        dot += d_e[l][2] * u.z;
        uu += u.z * u.z;
        float feq = d_w[l] * (density - 1.5*uu + 3.*dot + 4.5*dot*dot);
        d_cellsCur[index(i,j,k,l)] =
                d_omega * feq + (1.0-d_omega) * d_cellsCur[index(i,j,k,l)];
    }
}
__global__ void streamCuda(float *d_cellsCur, float *d_cellsLast, char *d_flags) {
    // get the current thread position
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;

    // in case we have no periodic boundaries, the threads on the edges don't have anything to do
    if (!PeriodicBoundaries) {
        if (i==0 || i==blockDim.x-1 || j==0 || j==blockDim.y-1 || k==0 || k==gridDim.x-1)
            return;
    }

    for(int l=0; l<Q; ++l) {
        const int inv = d_invDir[l];
        const int si = i+d_e[inv][0];
        const int sj = j+d_e[inv][1];
        const int sk = k+d_e[inv][2];
        if(d_flags[index(si,sj,sk)] == CellNoSlip) {
            // reflect at NoSlip cell
            d_cellsCur[index(i,j,k,l)] = d_cellsLast[index(i,j,k,inv)];
        }
        else {
            // update from neighbours
            d_cellsCur[index(i,j,k,l)] = d_cellsLast[index(si,sj,sk,l)];
        }
    }
}

__global__ void analyzeCuda(float *d_cellsCur, char *d_flags, float *d_density, float3 *d_u, float3 *d_velocity) {
    // get the current thread position
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;

    // compute density and velocity in cell
    float density = 0.f;
    float3 u = make_float3(0.f, 0.f, 0.f);
    if(d_flags[index(i,j,k)] == CellNoSlip) {
        density = 1.;
    }
    else {
        for(int l=0; l<Q; ++l) {
            const float weight = d_cellsCur[index(i,j,k,l)];
            density += weight;
            u.x += d_e[l][0] * weight;
            u.y += d_e[l][1] * weight;
            u.z += d_e[l][2] * weight;
        }
    }

    d_density[index(i,j,k)] = density;
    d_u[index(i,j,k)] = u;
}

__global__ void minMaxCuda() { }

//! we need some kind of initialization of our device
void LBMD3Q19::initializeCuda() {
    // get some space for our arrays
    cudaMalloc((void **) &d_flags, sizeof(char) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_velocity, sizeof(float3) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_u, sizeof(float3) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_density, sizeof(float) * m_width * m_height * m_depth);
    cudaMalloc((void **) &d_cells[0], sizeof(float) * m_width * m_height * m_depth * Q);
    cudaMalloc((void **) &d_cells[1], sizeof(float) * m_width * m_height * m_depth * Q);

    // use cpyToSymbol for known sizes (LEGACY CODE - WORKS ONLY WITH CUDA <= 5.5)
    cudaMemcpyToSymbol(d_w, w.w, sizeof(float)*Q);
    cudaMemcpyToSymbol(d_e, e, sizeof(int)*D*Q);
    cudaMemcpyToSymbol(d_invDir, invDir, sizeof(int)*Q);
}

//! collide implementation with CUDA
void LBMD3Q19::collideCuda() {
    collideCuda<<<dim3(m_width),dim3(m_height,m_depth)>>>(d_cells[m_current], d_flags, d_velocity);
}

//! streaming with CUDA
void LBMD3Q19::streamCuda() {
    streamCuda<<<dim3(m_width),dim3(m_height,m_depth)>>>(d_cells[m_current], d_cells[!m_current], d_flags);
}

//! compute densities and velocities with CUDA
void LBMD3Q19::analyzeCuda() {
    analyzeCuda<<<dim3(m_width),dim3(m_height,m_depth)>>>(d_cells[m_current], d_flags, d_density, d_u, d_velocity);
    // we need to copy back the analyzed data to the host
    cudaMemcpy(m_u, d_u, sizeof(float3) * m_width * m_height * m_depth, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_density, d_density, sizeof(float) * m_width * m_height * m_depth, cudaMemcpyDeviceToHost);
}

//! compute minimum and maximum density and velocity with CUDA
void LBMD3Q19::minMaxCuda() {
    minMaxCuda<<<dim3(m_width),dim3(m_height,m_depth)>>>();
}

//! very dumb function that copies cells back to host
void LBMD3Q19::cpCellsDeviceToHost() {
    cudaMemcpy(m_cells[m_current], d_cells[m_current], sizeof(float) * m_width * m_height * m_depth * Q, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_cells[!m_current], d_cells[!m_current], sizeof(float) * m_width * m_height * m_depth * Q, cudaMemcpyDeviceToHost);
}
//! free allocated data on device
void LBMD3Q19::freeCuda() {
    //! each malloc needs a free
    cudaMalloc(d_flags);
    cudaMalloc(d_velocity);
    cudaMalloc(d_u);
    cudaMalloc(d_density);
    cudaMalloc(d_cells[0]);
    cudaMalloc(d_cells[1]);
}

//! this needs to be done, each time we switch our settings
void LBMD3Q19::applyCuda() {
    //! copy data from host to device, the rest are constants which stay the same
    cudaMemcpy(d_flags, m_flags, m_width * m_height * m_depth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity, m_velocity, sizeof(float3) * m_width * m_height * m_depth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells[m_current], m_cells[m_current], sizeof(float) * m_width * m_height * m_depth * Q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells[!m_current], m_cells[!m_current], sizeof(float) * m_width * m_height * m_depth * Q, cudaMemcpyHostToDevice);
    //! omega can be changed, too
    cudaMemcpyToSymbol(d_omega, &m_omega, sizeof(float));
}

//! http://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__THREAD_g6e0c5163e6f959b56b6ae2eaa8483576.html
void LBMD3Q19::syncCuda() {
    cudaThreadSynchronize()
}
