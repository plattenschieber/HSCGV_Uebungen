/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <cfloat>
#include <cmath>
#include "geoquadric.h"

// Description:
// Constructor.
 __host__ __device__ GeoQuadric::GeoQuadric()
: m_a(0), m_b(0), m_c(0)
, m_d(0), m_e(0), m_f(0)
, m_g(0), m_h(0), m_j(0)
, m_k(0)
{
   m_properties = NULL;
}

// Description:
// explicit parametrization
 __host__ __device__ GeoQuadric::GeoQuadric(double na, double nb, double nc, double nd, double ne,
      double nf, double ng, double nh, double nj, double nk)
: m_a(na), m_b(nb), m_c(nc)
, m_d(nd), m_e(ne), m_f(nf)
, m_g(ng), m_h(nh), m_j(nj)
, m_k(nk)
{
   m_properties = NULL;
}

// Description:
// Destructor.
 __host__ __device__ GeoQuadric::~GeoQuadric()
{
}


// Description:
// Return the normal vector at point v of this surface
Vec3d __host__ __device__
GeoQuadric::getNormal(const Vec3d &v) const
{
   Vec3d tmpvec( (v | Vec3d(2*m_a,m_b,m_c)) + m_d,
         (v | Vec3d(m_b,2*m_e,m_f)) + m_g,
         (v | Vec3d(m_c,m_f,2*m_h)) + m_j );
   return tmpvec.getNormalized();
}

// Description:
// Compute intersection point on this surface
// return distance to nearest intersection found or -1
double __host__ __device__
GeoQuadric::intersect(const Ray &ray) const
{
   double t = -1.0, acoef, bcoef, ccoef, root, disc;

   acoef = Vec3d( (ray.direction() | Vec3d(m_a,m_b,m_c)),
         m_e * ray.direction()[1] + m_f * ray.direction()[2],
         m_h * ray.direction()[2])
      | ray.direction();

   bcoef =   (Vec3d(m_d,m_g,m_j) | ray.direction())
      + (ray.origin() | Vec3d((ray.direction() | Vec3d(2*m_a,m_b,m_c)),
               (ray.direction() | Vec3d(m_b,2*m_e,m_f)),
               (ray.direction() | Vec3d(m_c,m_f,2*m_h))));

   ccoef = (ray.origin() | Vec3d((Vec3d(m_a,m_b,m_c) | ray.origin()) + m_d,
            m_e * ray.origin()[1] + m_f * ray.origin()[2] + m_g,
            m_h * ray.origin()[2] + m_j))
      + m_k;

   if (acoef != 0.0) {
      disc = bcoef * bcoef - 4.0 * acoef * ccoef;
      if (disc > -Vec3d::getEpsilon()) {
         root = sqrt( disc );
         t = ( -bcoef - root ) / ( acoef + acoef );
         if (t < 0.0) 
            t = ( -bcoef + root ) / ( acoef + acoef );
      }
   }
   return (((Vec3d::getEpsilon() * 10.0) < t) ? t : -1.0);
}
