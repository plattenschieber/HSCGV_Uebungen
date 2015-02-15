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
    //TODO
}

// Description:
// Return the normal vector of a specific polygon
Vec3d
GeoPolygon::getNormal(const int nPoly) const
{
    Vec3d a = m_polygons.at(nPoly).at(2) - m_polygons.at(nPoly).at(0);
    Vec3d b = m_polygons.at(nPoly).at(1) - m_polygons.at(nPoly).at(0);
    // vector product
    return (a^b).getNormalized();
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
double
GeoPolygon::intersect(const Ray &ray, const int nPoly) const
{
    std::vector<int> p = m_polygons.at(nPoly);
    Vec3d edge, vp;
    Vec3d origin = ray.origin();
    Vec3d dir = ray.direction();
    double t, si;

    Vec3d norm = getNormal(nPoly);
    double nDir = norm | dir;
    double d = norm | p.at(0);


    if(nDir != 0.0)
    {
        /*Determine where the intersection is */
        si = norm | origin;
        t = (d - si) / nDir;
        Vec3d point = origin + dir*t;
        if ((t > Vec3d::getEpsilon()))
        {
            unsigned length = p.size();

            /* See if intersection is on polygon edge */
            for (unsigned i = 0; i < length - 1; i++)
            {
                edge = p.at(i+1) - p.at(i);
                vp = point - p.at(i);
                dir = edge^vp;
                if((vp | dir) < 0)
                {
                    return -1.;
                }
            }

            /* check the last edge */
            edge = p.at(length-1) - p.at(0);
            vp = point - p.at(0);
            dir = edge^vp;
            if((vp | dir) < 0)
            {
                return -1.;
            }

            /* see if intersection lies inside the intersection of all spanned planes */
            for (unsigned i = 1; i < length - 1; i++) {
                edge = p.at(i+1) - p.at(i);
                vp = point - p.at(i);
                dir = p.at(i-1) - p.at(i);
                if((vp | dir) < 0)
                {
                    return -1.;
                }

            }

            /* handle very first plane */
            edge = p.at(length-1) - p.at(0);
            vp = point - p.at(0);
            dir = p.at(1) - p.at(0);
            if((vp | dir) < 0)
            {
                return -1.;
            }

            /* and the last plane */
            edge = p.at(0) - p.at(1);
            vp = point - p.at(1);
            dir = p.at(2) - p.at(1);
            if((vp | dir) < 0)
            {
                return -1.;
            }

            // all checks survived, the point v is definitely inside this polygon
            return t;
        }
    }
    // nDir is 0!
    return -1.;
}
