/*
 * Created by Jeronim Morina <morina@jeronim.de>
 *
 * File: geopolygon.cpp
 *   Declaration of polygonal surfaces
 */
#include <cfloat>
#include <cmath>
#include "geopolygon.h"

// Description:
// Constructor.
GeoPolygon::GeoPolygon()
{
   m_properties = NULL;
}

// Description:
// add a new polygonal surface with a list of all vertices
GeoPolygon::GeoPolygon(std::vector<Vec3d> vertices, std::vector<std::vector<int> > polygons)
: m_vertices(vertices), m_polygons(polygons)
{
   m_properties = NULL;
}

// Description:
// Destructor.
GeoPolygon::~GeoPolygon()
{
    m_polygons.clear();
    m_vertices.clear();
}

// Description:
// Return the normal vector at point v of this surface
Vec3d
GeoPolygon::getNormal(const Vec3d &v) const
{
    Vec3d a, b, norm;
    v.getEpsilon();
//    a = p->points.at(2) - p->points.at(0);
//    b = p->points.at(1) - p->points.at(0);
    // vector product
    norm = a^b;
    return norm.getNormalized();
}

// Description:
// Compute intersection point on this surface
// return distance to nearest intersection found or -1
double
GeoPolygon::intersect(const Ray &ray) const
{
   double t = -1.0;
   ray.direction();
   return (((Vec3d::getEpsilon() * 10.0) < t) ? t : -1.0);
}
