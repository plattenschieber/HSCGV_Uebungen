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
      /*! Construct a polygonal surface consisting of a list of vertices and several connected polygons */
      GeoPolygon(std::vector<Vec3d> vertices, std::vector<std::vector<int> > polygons);
      //! Delete a polygon.
      virtual ~GeoPolygon();

      //! Compute surface normal on the polygon in the point v.
      virtual Vec3d  getNormal(const Vec3d &v) const;
      //! Compute intersection of ray with polygon.
      virtual double intersect(const Ray &r) const;

   private:
      //! parameters of the equation describing the polygon.
      std::vector<Vec3d> m_vertices;
      std::vector<std::vector<int> > m_polygons;
};

#endif // GEOPOLYGON_H
