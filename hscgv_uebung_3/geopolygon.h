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
      //! Construct trivial \b polygon.
      GeoPolygon();
      //! Construct a polygonal surface consisting of a list of vertices and several connected polygons
      GeoPolygon(std::vector<Vec3d> vertices, std::vector<std::vector<int> > polygons);
      //! Delete a polygon.
      virtual ~GeoPolygon();

      //! Compute surface normal on the polygon in the point \b v.
      /*! since we call this method only upon positive intersection, \b v is indeed a
       * point on the surface. This method finds out in which polygon it lies and returns
       * its easy to compute normal. In case that \b v lies on an edge or a vertex,
       * we take the first found polygon. \b Regard that this should be changed in
       * case there are too many artefacts.
       */
      virtual Vec3d  getNormal(const Vec3d &v) const;
      //! Compute intersection of ray with polygon.
      virtual double intersect(const Ray &r) const;

   private:
      //! All vertices belonging to this polygonal surface
      std::vector<Vec3d> m_vertices;
      //! This polygonal surface consists of several polygons
      /*! Each polygon is saved as a list of indices, pointing into \b m_vertices */
      std::vector<std::vector<int> > m_polygons;

      //! This method returns the normal to nPoly-th polygon
      Vec3d getNormal(const int nPoly) const;
};

#endif // GEOPOLYGON_H
