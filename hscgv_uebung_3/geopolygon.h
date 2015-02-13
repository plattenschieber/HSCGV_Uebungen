/*
 * Created by Jeronim Morina <morina@jeronim.de>
 *
 * File: geopolygon.h
 *   Declaration of polygonal surfaces
 */
#ifndef GEOPOLYGON_H
#define GEOPOLYGON_H

#include "geoobject.h"
#include "ray.h"


//! A polygon surface
/*! This type implements a special type of geometric objects,
  polygon surfaces given in implicit form.
 */
class GeoPolygon : public GeoObject
{
   public:
      // CONSTRUCTORS
      //! Construct trivial polygon.
      GeoPolygon();
      /*! Construct polygon consisting of the points (x,y,z) satisfying
        na x^2+ nb x y + nc x z + nd x + ne y^2 + nf y z + ng y + nh z^2 + nj z + nk = 0
       */
      GeoPolygon(double na, double nb, double nc, double nd, double ne,
            double nf, double ng, double nh, double nj, double nk);
      //! Delete a polygon.
      virtual ~GeoPolygon();

      //! Compute surface normal on the polygon in the point v.
      virtual Vec3d  getNormal(const Vec3d &v) const;
      //! Compute intersection of ray with polygon.
      virtual double intersect(const Ray &r) const;

   private:
      //! parameters of the equation describing the polygon.
      double m_a,m_b,m_c,m_d,m_e,m_f,m_g,m_h,m_j,m_k;
};

#endif // GEOPOLYGON_H
