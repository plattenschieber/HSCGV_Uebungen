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
  polygon surfaces given in a vertex-index-list form
 */
class GeoPolygon : public GeoObject
{
   public:
      // CONSTRUCTORS
      //! Construct trivial polygon.
      GeoPolygon();
      //! Construct a polygonal surface consisting of a list of vertices and several connected polygons
      GeoPolygon(std::vector<Vec3d> vertices, std::vector<std::vector<int> > polygons);
      //! Delete a polygon.
      virtual ~GeoPolygon();

      //! Compute surface normal on the polygon in the point v.
      /*! v is indeed a point on the surface, so this method finds out in
       * which polygon it lies and then return its easy to compute normal */
      virtual Vec3d  getNormal(const Vec3d &v) const;
      //! Compute intersection of ray with polygon.
      virtual double intersect(const Ray &r) const;

   private:
      //! All vertices belonging to this polygonal surface
      std::vector<Vec3d> m_vertices;
      //! This polygonal surface consists of several polygons
      /*! Each polygon is saved as a list of indices, pointing into \b m_vertices */
      std::vector<std::vector<int> > m_polygons;
};

#endif // GEOPOLYGON_H
