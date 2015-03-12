/** \file
 * \brief 3D flow simulation with Lattice Boltzmann Method
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "lbm.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>

LBMD3Q19::LBMD3Q19(int width, int height, int depth)
: m_current(0)
, m_omega(1.0)
, m_width(width)
, m_height(height)
, m_depth(depth)
, m_step(0)
, m_useCuda(false)
{
    assert(D==3);
    assert(Q==19);

    switch(D)
    {
        case 3:
            switch(Q)
            {
                case 19:
                    typedef Velocity V;

                    e[0] = V( 0, 0, 0);
                    invDir[0] = 0;

                    e[1] = V( 0, 1, 0);
                    invDir[1] = 2;
                    e[2] = V( 0,-1, 0);
                    invDir[2] = 1;
                    e[3] = V( 1, 0, 0);
                    invDir[3] = 4;
                    e[4] = V(-1, 0, 0);
                    invDir[4] = 3;
                    e[5] = V( 0, 0, 1);
                    invDir[5] = 6;
                    e[6] = V( 0, 0,-1);
                    invDir[6] = 5;

                    e[7] = V( 1, 1, 0);
                    invDir[7] = 10;
                    e[8] = V(-1, 1, 0);
                    invDir[8] = 9;
                    e[9] = V( 1,-1, 0);
                    invDir[9] = 8;
                    e[10]= V(-1,-1, 0);
                    invDir[10] = 7;

                    e[11]= V( 0, 1, 1);
                    invDir[11] = 14;
                    e[12]= V( 0, 1,-1);
                    invDir[12] = 13;
                    e[13]= V( 0,-1, 1);
                    invDir[13] = 12;
                    e[14]= V( 0,-1,-1);
                    invDir[14] = 11;

                    e[15]= V( 1, 0, 1);
                    invDir[15] = 18;
                    e[16]= V( 1, 0,-1);
                    invDir[16] = 17;
                    e[17]= V(-1, 0, 1);
                    invDir[17] = 16;
                    e[18]= V(-1, 0,-1);
                    invDir[18] = 15;

                    break;
            }
            break;
    }

    // check if plausible values have been used
    for(int i=0; i<Q; ++i)
    {
        const int inv = invDir[i];
        assert(invDir[inv] == i);
        int l2 = 0;
        for(int j=0; j<D; ++j)
        {
            l2 += e[i][j] * e[i][j];
            assert(e[i][j] == -e[inv][j]);
            assert(e[i][j] == 0 || e[i][j]*e[i][j] == 1);
        }
        assert(l2==0 || l2==1 || l2==2);
    }

    m_flags = new Flag[m_width * m_height * m_depth];
    memset(m_flags, '\0', sizeof(Flag)*m_width*m_height*m_depth);
    m_velocity = new Vector[m_width * m_height * m_depth];
    m_u = new Vector[m_width * m_height * m_depth];
    m_density = new Scalar[m_width * m_height * m_depth];
    for(int i=0; i<2; ++i)
        m_cells[i] = new Scalar[m_width * m_height * m_depth * Q];

    for(int k=0; k<m_depth; ++k)
    {
        for(int j=0; j<m_height; ++j)
        {
            for(int i=0; i<m_width; ++i)
            {
                m_flags[index(i,j,k)] = CellFluid;
                for(int l=0; l<Q; ++l)
                {
                    m_cells[0][index(i,j,k,l)] = w[l];
                    m_cells[1][index(i,j,k,l)] = w[l];
                }
            }
        }
    }

    if(!PeriodicBoundaries)
        setNoSlipBorders();
    //! do the required step to use the CUDA acceleration
    initializeCuda();
}

void LBMD3Q19::setOmega(double omega)
{
    assert(omega >= 0. && omega <= 2.);
    m_omega = omega;
}

void LBMD3Q19::setNoSlipBorders()
{
    for(int k=0; k<m_depth; ++k)
    {
        for(int j=0; j<m_height; ++j)
        {
            for(int i=0; i<m_width; ++i)
            {
                if(i==0 || j==0 || k==0 || k==m_depth-1 || j==m_height-1 || i==m_width-1)
                    setNoSlip(i, j, k);
            }
        }
    }
}

void LBMD3Q19::setNoSlipBlock(int fx, int fy, int fz, int tx, int ty, int tz)
{
    assert(fx >= 0);
    assert(fy >= 0);
    assert(fz >= 0);
    assert(tx <= m_width);
    assert(ty <= m_height);
    assert(tz <= m_depth);

    for(int k=fz; k<tz; ++k)
    {
        for(int j=fy; j<ty; ++j)
        {
            for(int i=fx; i<tx; ++i)
            {
                setNoSlip(i, j, k);
            }
        }
    }
}

LBMD3Q19::~LBMD3Q19()
{

    for(int i=0; i<2; ++i)
        delete[] m_cells[i];
    delete[] m_flags;
    delete[] m_u;
    delete[] m_density;

    //! free the device
    freeCuda();
}

void LBMD3Q19::apply()
{
    //! this has to be done at least one time, when using acceleration
    applyCuda();
}

int LBMD3Q19::getDimension(int c) const
{
    switch(c)
    {
        case 0:
            return m_width;
        case 1:
            return m_height;
        case 2:
            return m_depth;
    }

    return 0;
}

void LBMD3Q19::setVelocity(int i, int j, int k, const Vector &u)
{
    m_flags[index(i,j,k)] = CellVelocity;
    m_velocity[index(i,j,k)] = u;
}

void LBMD3Q19::setNoSlip(int i, int j, int k)
{
    m_flags[index(i,j,k)] = CellNoSlip;
}

void LBMD3Q19::step()
{
    m_current = !m_current;
    if (!m_useCuda) {
#       pragma omp parallel
        {
            streamCpu();
            collideCpu();
        }
    }
    //! use CUDA to stream and collide
    else {
        streamCuda();
        collideCuda();
    }
    ++m_step;
}

void LBMD3Q19::analyze()
{
    if (!m_useCuda) {
        analyzeCpu();
    }
    //! use CUDA to anlyze the data
    else {
        analyzeCuda();
        // not yet implemented minMaxCuda();
    }
    minMaxCpu();
}

int LBMD3Q19::getStep() const
{
    return m_step;
}


LBMD3Q19::Scalar LBMD3Q19::density(int i, int j, int k) const
{
    return m_density[index(i,j,k)];
}

LBMD3Q19::Vector LBMD3Q19::velocity(int i, int j, int k) const
{
    return m_u[index(i,j,k)];
}

void LBMD3Q19::streamCpu()
{
#  pragma omp for schedule(static)
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

void LBMD3Q19::analyzeCpu()
{
#  pragma omp parallel for schedule(static)
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

void LBMD3Q19::minMaxCpu()
{
    // reset minium and maximum values
    m_minDensity = 1000.;
    m_maxDensity = 0.;
    m_maxVelocity2 = 0.;

#  pragma omp for schedule(static)
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

void LBMD3Q19::collideCpu()
{
#  pragma omp for schedule(static)
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

LBMD3Q19::Scalar LBMD3Q19::maxDensity() const
{
    return m_maxDensity;
}

LBMD3Q19::Scalar LBMD3Q19::minDensity() const
{
    return m_minDensity;
}

LBMD3Q19::Scalar LBMD3Q19::maxVelocity() const
{
    return sqrtf(m_maxVelocity2);
}

void LBMD3Q19::useCuda(bool enable)
{
    //! copy cells back to host, if CUDA was enabled and we want to disable GPU
    //! otherwise, we would copy data back and forth
    if(m_useCuda && enable == false)
        cpCellsDeviceToHost();
    //! change state
    m_useCuda = enable;
    //! move cells back to device if we enabled GPU acceleration and update device data
    if (m_useCuda)
        applyCuda();
}

void LBMD3Q19::sync() const
{
    if (m_useCuda)
        syncCuda();
}
