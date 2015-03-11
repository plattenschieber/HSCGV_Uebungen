/** \file
 * \brief 3D flow simulation with Lattice Boltzmann Method
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef LBM_H
#define LBM_H

#include <cstdlib>


//#define INNER_INDEX_DISTRIBUTION

//! LBM simulation on a 3-dimensional grid using a neighbourhood of size 19
class LBMD3Q19
{
    public:
        // constants
        //! 0 = no periodic boundaries, 1 = periodic boundaries -- non-periodic is faster
        static const int PeriodicBoundaries = 0;
        //! dimensionaly of domain, has to be 3
        static const int D = 3;
        //! size of neighbourhood of a cell, has to be 19
        static const int Q = 19;

        //! has to be float for current CUDA implementation
        typedef float Scalar;
        //! has to be char for curent CUDA implementation
        typedef char Flag;

        //! cell types
        enum CellFlags
        {
            //! a wet cell
            CellFluid = 0,
            //! a wall cell, flow bounces back
            CellNoSlip,
            //! fixed velocity cell
            CellVelocity
        };

        // types
        //! a scalar vector
        struct Vector
        {
            //! vector data
            Scalar e[D];
            //! construct zero vector
            Vector()
            {
                for(int c=0; c<D; ++c)
                    e[c] = 0.;
            }
            //! construct from an array of scalars
            Vector(Scalar *s)
            {
                for(int c=0; c<D; ++c)
                    e[c] = s[c];
            }
            //! construct from three scalars
            Vector(Scalar x, Scalar y, Scalar z)
            {
                e[0] = x;
                e[1] = y;
                e[2] = z;
            }
            //! accessor for vector components
            Scalar &operator[](int i) { return e[i]; }
            //! const accessor for vector components
            const Scalar &operator[](int i) const { return e[i]; }
        };

        //! density for particle distribution function
        struct Distribution
        {
            //! distribution data
            Scalar w[Q];
            //! construct initial distribution for D3Q19 LBM simulation
            Distribution()
            {
                // weights
                w[0] = 1./3.;
                for(int i=1; i<7; ++i)
                    w[i] = 1./18.;
                for(int i=7; i<19; ++i)
                    w[i] = 1./36.;

            }
            //! accessor for distribution components
            Scalar &operator[](int i) { return w[i]; }
            //! const accessor for distribution components
            const Scalar &operator[](int i) const { return w[i]; }
        };

        //! discrete velocity corresponding to a particle distribution entry
        struct Velocity
        {
            //! velocity components
            int e[D];
            //! construct from components
            Velocity(int x=0, int y=0, int z=0)
            {
                e[0] = x;
                if(D > 1)
                    e[1] = y;
                if(D > 2)
                    e[2] = z;
            }
            //! access velocity component
            int &operator[](int i) { return e[i]; }
            //! constant accessor for velocity components
            const int &operator[](int i) const { return e[i]; }
        };

        // methods
        //! initiate simulation on a grid of size width x height x depth
        LBMD3Q19(int width, int height, int depth);
        //! dtor
        ~LBMD3Q19();

        //! apply changes to simulation state
        void apply();

        //! return size of simulation grid in dimension index
        int getDimension(int index) const;
        //! get number of simulation steps completed
        int getStep() const;

        //! switch GPU acceleration on and off
        void useCuda(bool enable);

        //! compute densities and velocities
        void analyze();
        //! wait for in-flight operations to complete (in order to get valid timing values)
        void sync() const;
        //! return most recently computed density
        Scalar density(int i, int j, int k) const;
        //! return most recently computed velocity
        Vector velocity(int i, int j, int k) const;

        //! set relaxation coefficient
        void setOmega(double omega);

        //! set a fixed velocity for a cell
        void setVelocity(int i, int j, int k, const Vector &u);

        //! make a NoSlip cell
        void setNoSlip(int i, int j, int k);
        //! make all border cells NoSlip
        void setNoSlipBorders();
        //! set a block of cells to NoSlip
        void setNoSlipBlock(int fx, int fy, int fz, int tx, int ty, int tz);

        //! perform a simulation step
        void step();

        //! return maximum density as computed by analyze
        Scalar maxDensity() const;
        //! return minimum density as computed by analyze
        Scalar minDensity() const;
        //! return maximum absolute velocity as computed by analyze
        Scalar maxVelocity() const;


    private:
        //! weights of discrete speeds
        Distribution w;
        //! discrete velocities to neighbour cells
        Velocity e[Q];
        //! index of inverse velocity
        int invDir[Q];

        //! cell types
        Flag *m_flags;
        //! prescribed velocity for Velocity cells
        Vector *m_velocity;
        //! simulated velocity
        Vector *m_u;
        //! simulated density
        Scalar *m_density;

        //! two arrays of distributions that are updated alternating, each indexed by 4 coordinates
        Scalar *m_cells[2];
        //! current distribution array
        int m_current;
        //! relaxation coefficient
        Scalar m_omega;
        //! width of simulation grid
        int m_width;
        //! height of simulation grid
        int m_height;
        //! depth of simulation grid
        int m_depth;
        //! current simulation step
        int m_step;
        //! minimum density at last analysis step
        Scalar m_minDensity;
        //! maximum density at last analysis step
        Scalar m_maxDensity;
        //! maximum velocity at last analysis step
        Scalar m_maxVelocity2;
        //! flag to indicate if CUDA is to be used
        bool m_useCuda;

        //! linearize 3-dimensional array
        size_t index(int i, int j, int k) const
        {
            if(PeriodicBoundaries)
            {
                i = (i+m_width)%m_width;
                j = (j+m_height)%m_height;
                k = (k+m_depth)%m_depth;
            }
            return i + m_width*(j + m_height*size_t(k));
        }

        //! linearize 4-dimensional array
        size_t index(int i, int j, int k, int l) const
        {
            if(PeriodicBoundaries)
            {
                i = (i+m_width)%m_width;
                j = (j+m_height)%m_height;
                k = (k+m_depth)%m_depth;
            }
#ifdef INNER_INDEX_DISTRIBUTION
            return l + Q*(i + m_width*(j + m_height*size_t(k)));
#else
            return i + m_width*(j + m_height*(size_t(k) + m_depth*l));
#endif
        }

        //! perform collision step on CPU
        void collideCpu();
        //! perform streaming step on CPU
        void streamCpu();

        //! compute densities and velocities on CPU
        void analyzeCpu();
        //! compute minimum and maximum density and velocity on CPU
        void minMaxCpu();

        // ------ CUDA

        // we need a CUDA pendant for every allocated datatype
        //! cell types
        char *d_flags;
        //! prescribed velocity for Velocity cells
        float3 *d_velocity;
        //! simulated velocity
        float3 *d_u;
        //! simulated density
        float *d_density;
        //! two arrays of distributions that are updated alternating, each indexed by 4 coordinates
        float *d_cells[2];

        // function pendants in CUDA
        //! we need some kind of initialization of our device
        void intializeCuda();

        //! collide implementation with CUDA
        void collideCuda();
        //! streaming with CUDA
        void streamCuda();

        //! compute densities and velocities with CUDA
        void analyzeCuda();
        //! compute minimum and maximum density and velocity with CUDA
        void minMaxCpu();
};

#endif
