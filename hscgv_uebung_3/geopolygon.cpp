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
    // find out where v lies
    for (unsigned poly=0; poly<m_polygons.size(); poly++)
    {
        double minRange = DBL_MAX;
        int    minRangeIndex = -1;
        // to use pnpoly appropriately we need to delete the coordinate
        // with the smallest range due to numerical stability
        // compare all x,y,z seperately
        for (unsigned i=0; i<3; i++)
        {
            for (unsigned j=0; j<m_polygons.at(poly).size(); j++)
            double maxItem = -DBL_MAX, minItem = DBL_MAX;
            {
                maxItem = MAX(maxItem, m_vertices[m_polygons[poly][j]][i]);
                minItem = MIN(minItem, m_vertices[m_polygons[poly][j]][i]);
            }
            // compare if ... cordinate has smaller range
            if (maxItem-minItem < minRange)
            {
                minRange = maxItem - minItem;
                minRangeIndex = i;
            }
        }
        // get rid of ... cordinate
        // cordinate vector cvector = {0, 1, x2x} or {x0x, 1, 2},..
        int cVector[] = {0,2};

        // fill vertx and verty
        float *vertx; //= m_polygons.at(indexOuter)[cVector[0]];
        float *verty; //= m_polygons.at(indexOuter)[cVector[0]];
        for (unsigned i=0; i<m_polygons.at(indexOuter).size(); i++)
        {
              // fill them all
            ;
        }

        // check if v lies in this projected polygon
        if (pnpoly(m_polygons.at(indexOuter).size(), vertx, verty, v[cVector[0]], v[cVector[1]]))
        {
            return getNormal(indexOuter);
        }
    }

    // this should never happen!
    return getNormal(0);
}

// Description:
// Return the normal vector of a specific polygon
Vec3d
GeoPolygon::getNormal(const int nPoly) const
{
    Vec3d a = m_polygons[nPoly][2] - m_polygons[nPoly][0];
    Vec3d b = m_polygons[nPoly][1] - m_polygons[nPoly][0];
    // vector product
    return (a^b).getNormalized();
}

// Description:
// Copyright (c) 1970-2003, Wm. Randolph Franklin
int
GeoPolygon::pnpoly(int nvert, float *vertx, float *verty, float testx, float testy) const
{
    int i, j, c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) &&
             (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
            c = !c;
    }
    return c;
}


// Description:
// Compute intersection point on this surface
// return distance to nearest intersection found or -1
double
GeoPolygon::intersect(const Ray &ray) const
{
    double temp, lastTemp=DBL_MAX;
    for (unsigned i=0; i<m_polygons.size(); i++)
    {
        temp = intersect(ray, i);
        if (temp >= 0 && temp < lastTemp) lastTemp = temp;
    }
    return (lastTemp<DBL_MAX)?lastTemp:-1;
}

double
GeoPolygon::intersect(const Ray &ray, const int nPoly) const
{
    std::vector<int> p = m_polygons[nPoly];
    Vec3d edge, vp;
    Vec3d origin = ray.origin();
    Vec3d dir = ray.direction();
    double t, si;

    Vec3d norm = getNormal(nPoly);
    double nDir = norm | dir;
    double d = norm | p[0];


    if(nDir != 0.0)
    {
        // determine where the intersection is
        si = norm | origin;
        t = (d - si) / nDir;
        Vec3d point = origin + dir*t;
        if ((t > Vec3d::getEpsilon()))
        {
            unsigned length = p.size();

            // see if intersection is on polygon edge
            for (unsigned i = 0; i < length - 1; i++)
            {
                edge = p[i+1] - p[i];
                vp = point - p[i];
                dir = edge^vp;
                if((vp | dir) < 0)
                {
                    return -1.;
                }
            }

            // check the last edge
            edge = p[length-1] - p[0];
            vp = point - p[0];
            dir = edge^vp;
            if((vp | dir) < 0)
            {
                return -1.;
            }

            // see if intersection lies inside the intersection of all spanned planes
            for (unsigned i = 1; i < length - 1; i++) {
                edge = p[i+1] - p[i];
                vp = point - p[i];
                dir = p[i-1] - p[i];
                if((vp | dir) < 0)
                {
                    return -1.;
                }

            }

            // handle very first plane
            edge = p[length-1] - p[0];
            vp = point - p[0];
            dir = p[1] - p[0];
            if((vp | dir) < 0)
            {
                return -1.;
            }

            // and the last plane
            edge = p[0] - p[1];
            vp = point - p[1];
            dir = p[2] - p[1];
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
