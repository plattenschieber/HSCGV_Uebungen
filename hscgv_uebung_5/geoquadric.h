/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: geoquadric.h
 *   Declaration of quadric surfaces
 */
#ifndef GEOQUADRIC_HH
#define GEOQUADRIC_HH

#include "geoobject.h"
#include "ray.h"


//! A quadric surface
/*! This type implements a special type of geometric objects,
  quadric surfaces given in implicit form.
 */
class GeoQuadric : public GeoObject
{
   public:
      // CONSTRUCTORS
      //! Construct trivial quadric.
      GeoQuadric();
      /*! Construct quadric consisting of the points (x,y,z) satisfying
        na x^2+ nb x y + nc x z + nd x + ne y^2 + nf y z + ng y + nh z^2 + nj z + nk = 0
       */   
      GeoQuadric(double na, double nb, double nc, double nd, double ne,
            double nf, double ng, double nh, double nj, double nk);
      //! Delete a quadric.
      virtual ~GeoQuadric();

      //! Compute surface normal on the quadric in the point v.
      virtual Vec3d  getNormal(const Vec3d &v) const;
      //! Compute intersection of ray with quadric.
      virtual double __host__ __device__ intersect(const Ray &r) const;

   private:
      //! parameters of the equation describing the quadric.
      double m_a,m_b,m_c,m_d,m_e,m_f,m_g,m_h,m_j,m_k;
};
#include "geoquadric.inl"

#endif /* GEOQUADRIC_HH */
