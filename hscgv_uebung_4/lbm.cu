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

__global__ collideCuda() {
    for(int k=1-PeriodicBoundaries; k<m_depth-1+PeriodicBoundaries; ++k)
    {
        for(int j=1-PeriodicBoundaries; j<m_height-1+PeriodicBoundaries; ++j)
        {
            for(int i=1-PeriodicBoundaries; i<m_width-1+PeriodicBoundaries; ++i)
            {
                // nothing to do for NoSlip cells
                const int flag = m_flags[index(i,j,k)];
                if (flag == CellNoSlip)
                    continue;

                // compute density and velocity in cell
                Scalar density = 0.0;
                Vector u;
                for(int l=0; l<Q; ++l)
                {
                    const Scalar weight = m_cells[m_current][index(i,j,k,l)];
                    density += weight;
                    for(int c=0; c<D; ++c)
                        u[c] += e[l][c] * weight;
                }

                // override velocity for Velocity cells
                if (flag == CellVelocity)
                {
                    u = m_velocity[index(i,j,k)];
                }

                // collision
                for(int l=0; l<Q; ++l)
                {
                    Scalar dot = 0.;
                    Scalar uu = 0.;
                    for(int c=0; c<D; ++c)
                    {
                        dot += e[l][c] * u[c];
                        uu += u[c] * u[c];
                    }
                    Scalar feq = w[l] * (density - 1.5*uu + 3.*dot + 4.5*dot*dot);
                    m_cells[m_current][index(i,j,k,l)] =
                            m_omega * feq + (1.0-m_omega) * m_cells[m_current][index(i,j,k,l)];
                }
            }
        }
    }
}
__global__ streamCuda() {
    for(int k=1-PeriodicBoundaries; k<m_depth-1+PeriodicBoundaries; ++k)
    {
        for(int j=1-PeriodicBoundaries; j<m_height-1+PeriodicBoundaries; ++j)
        {
            for(int i=1-PeriodicBoundaries; i<m_width-1+PeriodicBoundaries; ++i)
            {
                for(int l=0; l<Q; ++l)
                {
                    const int inv = invDir[l];
                    const int si = i+e[inv][0];
                    const int sj = j+e[inv][1];
                    const int sk = k+e[inv][2];
                    if(m_flags[index(si,sj,sk)] == CellNoSlip)
                    {
                        // reflect at NoSlip cell
                        m_cells[m_current][index(i,j,k,l)] = m_cells[!m_current][index(i,j,k,inv)];
                    }
                    else
                    {
                        // update from neighbours
                        m_cells[m_current][index(i,j,k,l)] = m_cells[!m_current][index(si,sj,sk,l)];
                    }
                }
            }
        }
    }
}

__global__ analyzeCuda()
{
    for(int k=0; k<m_depth; ++k)
    {
        for(int j=0; j<m_height; ++j)
        {
            for(int i=0; i<m_width; ++i)
            {
                // compute density and velocity in cell
                Scalar density = 0.0;
                Vector u;
                if(m_flags[index(i,j,k)] == CellNoSlip)
                {
                    density = 1.;
                }
                else
                {
                    for(int l=0; l<Q; ++l)
                    {
                        const Scalar weight = m_cells[m_current][index(i,j,k,l)];
                        density += weight;
                        for(int c=0; c<D; ++c)
                            u[c] += e[l][c] * weight;
                    }
                }

                m_density[index(i,j,k)] = density;
                m_u[index(i,j,k)] = u;
            }
        }
    }
}

__global__ minMaxCuda()
{
    // reset minium and maximum values
    m_minDensity = 1000.;
    m_maxDensity = 0.;
    m_maxVelocity2 = 0.;

    for(int k=0; k<m_depth; ++k)
    {
        for(int j=0; j<m_height; ++j)
        {
            for(int i=0; i<m_width; ++i)
            {
                const size_t idx = index(i,j,k);
                // nothing to do for NoSlip cells
                const int flag = m_flags[idx];
                if (flag == CellNoSlip)
                    continue;

                // store min and max values - we don't care for race conditions
                if(m_density[idx] < m_minDensity)
                    m_minDensity = m_density[idx];
                if(m_density[idx] > m_maxDensity)
                    m_maxDensity = m_density[idx];
                float v2 = 0.;
                for(int c=0; c<D; ++c)
                {
                    v2 += m_u[idx][c] * m_u[idx][c];
                }
                if(v2 > m_maxVelocity2)
                    m_maxVelocity2 = v2;
            }
        }
    }
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
    collideCuda<<<dim3(m_width,m_height),dim3(m_depth)>>();
}

//! streaming with CUDA
void LBMD3Q19::streamCuda() {
    streamCude<<<dim3(m_width,m_height),dim3(m_depth)>>();
}

//! compute densities and velocities with CUDA
void LBMD3Q19::analyzeCuda() {
    analyzeCuda<<<dim3(m_width,m_height),dim3(m_depth)>>();
}

//! compute minimum and maximum density and velocity with CUDA
void LBMD3Q19::minMaxCuda() {
    minMaxCuda<<<dim3(m_width,m_height),dim3(m_depth)>>();
}
