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

__global__ collideCuda(float *d_cellsCur, char *d_flags, float3 *d_velocity) {
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
        continue;

    // compute density and velocity in cell
    float density = 0.0;
    float3 u = make_float3(0., 0., 0.,);
    for(int l=0; l<Q; ++l) {
        const float weight = d_cellsCur[index(i,j,k,l)];
        density += weight;
        for(int c=0; c<D; ++c)
            u[c] += d_e[l][c] * weight;
    }

    // override velocity for Velocity cells
    if (flag == CellVelocity) {
        u = d_velocity[index(i,j,k)];
    }

    // collision
    for(int l=0; l<Q; ++l) {
        float dot = 0.;
        float uu = 0.;
        for(int c=0; c<D; ++c) {
            dot += d_e[l][c] * u[c];
            uu += u[c] * u[c];
        }
        float feq = d_w[l] * (density - 1.5*uu + 3.*dot + 4.5*dot*dot);
        d_cellsCur[index(i,j,k,l)] =
                d_omega * feq + (1.0-d_omega) * d_cellsCur[index(i,j,k,l)];
    }
}
__global__ streamCuda(float *d_cellsCur, float *d_cellsLast, char *d_flags) {
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

__global__ analyzeCuda(float *d_cellsCur, char *d_flags, float *d_density, float3 *d_u, float3 *d_velocity) {
    // get the current thread position
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;

    // compute density and velocity in cell
    float density = 0.0;
    float3 u = make_float3(0., 0., 0.,);
    if(d_flags[index(i,j,k)] == CellNoSlip) {
        density = 1.;
    }
    else {
        for(int l=0; l<Q; ++l) {
            const float weight = d_cellsCur[index(i,j,k,l)];
            density += weight;
            for(int c=0; c<D; ++c)
                u[c] += d_e[l][c] * weight;
        }
    }

    d_density[index(i,j,k)] = density;
    d_u[index(i,j,k)] = u;
}

__global__ minMaxCuda() {
    // get the current thread position
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;

    // reset minium and maximum values
    d_minDensity = 1000.;
    d_maxDensity = 0.;
    d_maxVelocity2 = 0.;

    const size_t idx = index(i,j,k);
    // nothing to do for NoSlip cells
    const int flag = d_flags[idx];
    if (flag == CellNoSlip)
        continue;

    // store min and max values - we don't care for race conditions
    if(d_density[idx] < d_minDensity)
        d_minDensity = d_density[idx];
    if(d_density[idx] > d_maxDensity)
        d_maxDensity = d_density[idx];
    float v2 = 0.;
    for(int c=0; c<D; ++c) {
        v2 += d_u[idx][c] * d_u[idx][c];
    }
    if(v2 > d_maxVelocity2)
        d_maxVelocity2 = v2;
}

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
    collideCuda<<<dim3(m_width),dim3(m_height,m_depth)>>(d_cells[m_current], d_flags, d_velocity);
}

//! streaming with CUDA
void LBMD3Q19::streamCuda() {
    streamCude<<<dim3(m_width),dim3(m_height,m_depth)>>>(d_cells[m_current], d_cells[!m_current], d_flags);
}

//! compute densities and velocities with CUDA
void LBMD3Q19::analyzeCuda() {
    analyzeCuda<<<dim3(m_width),dim3(m_height,m_depth)>>>(d_cells[m_current], d_flags, d_density, d_u, d_velocity);
}

//! compute minimum and maximum density and velocity with CUDA
void LBMD3Q19::minMaxCuda() {
    minMaxCuda<<<dim3(m_width),dim3(m_height,m_depth)>>>();
}
