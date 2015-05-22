/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <climits>
#include <cmath>
#include <cfloat>
#include "ray.h"
#include "param.h"
#include "types.h"

// Description:
// Constructor.
inline Ray::Ray( )
: m_origin()
, m_direction()
, m_depth(0)
, m_objList(NULL)
, m_lightList(NULL)
{
}

// Description:
// Copy-Constructor.
inline Ray::Ray( const Ray &r )
{
   // copy it ! initialization is not enough !
   m_origin    = r.m_origin;
   m_direction = r.m_direction;
   m_depth     = r.m_depth;
   m_objList   = r.m_objList;
   m_lightList = r.m_lightList;
}

// Description:
// Constructor with explicit parameters
inline Ray::Ray(const Vec3d &o, const Vec3d &d, unsigned int i, std::vector<GeoObject*> &ol, std::vector<LightObject*> &ll)
{
   // copy it ! initialization is not enough !
   m_origin    = o;
   m_direction = d;
   m_depth     = i;
   m_objList   = &ol;
   m_lightList = &ll;
}

// Description:
// Constructor with explicit parameters
inline __device__ Ray::Ray(const Vec3d o, const Vec3d d, unsigned int i, GeoObject *ol, LightObject *ll)
{
   // copy it ! initialization is not enough !
   m_origin    = o;
   m_direction = d;
   m_depth     = i;
   d_objList   = ol;
   d_lightList = ll;
}

// Description:
// Destructor.
inline Ray::~Ray( )
{
}

// Description:
// Determine color of this ray by tracing through the scene
inline const Color __host__
Ray::shade() const
{
   GeoObject *closest = NULL;
   double tMin = DBL_MAX;

   // find closest object that intersects
   for (std::vector<GeoObject*>::iterator iter = m_objList->begin();
         iter != m_objList->end();
         iter++) {

      double t = (*iter)->intersect(*this);
      if (0.0 < t && t < tMin) {
         tMin = t;
         closest = (*iter);
      }
   }

   // no object hit -> ray goes to infinity
   if (closest == NULL) {
      if (m_depth == 0) {
         return g_scene.picture.background; // background color
      }
      else {
         return Color(0.0);         // black
      }
   }
   else {
      // reflection
      Vec3d intersectionPosition(m_origin + (m_direction * tMin));
      Vec3d normal(closest->getNormal(intersectionPosition));
      Ray reflectedRay(intersectionPosition,
            m_direction.getReflectedAt(normal).getNormalized(),
            m_depth+1,*m_objList,*m_lightList);
      Color currentColor(0.0);

      // calculate lighting
      for (std::vector<LightObject*>::iterator iter = m_lightList->begin();
            iter != m_lightList->end();
            iter++) {

         // where is the lightsource ?
         Ray rayoflight(intersectionPosition, (*iter)->direction(), 0, *m_objList, *m_lightList);
         bool something_intersected = false;

         // where are the objects ?
         for (std::vector<GeoObject*>::iterator iter2 = m_objList->begin();
               iter2 != m_objList->end();
               iter2++) {

            double t = (*iter2)->intersect(rayoflight);
            if (t > 0.0) {
               something_intersected = true;
               break;
            }

         } // for all obj

         // is it visible ?
         if (! something_intersected)
            currentColor += shadedColor((*iter), reflectedRay, normal, closest);

      } // for all lights

      // recurse ?
      if (m_depth < 5)
         currentColor += reflectedRay.shade() * closest->mirror();

      return Color(currentColor);
   }
}
// Description:
// Determine color of this ray by tracing through the scene
inline const Color __device__
Ray::cudaShade(GeoObject* m_objListCuda, int objListSize, LightObject* m_lightListCuda, int lightListSize) const
{
    Color currentColor(0.0);
    for (int i=0; i<5; i++) {
        GeoObject *closest = NULL;
        double tMin = DBL_MAX;

        // find closest object that intersects
        for (int j=0; j<objListSize; j++)
        {
            double t = m_objListCuda[j].intersect(*this);
            if (0.0 < t && t < tMin) {
                tMin = t;
                closest = &m_objListCuda[j];
            }
        }

        // no object hit -> ray goes to infinity
        if (closest == NULL) {
            if (m_depth == 0) {
//                return g_sceneCuda.picture.background; // background color
            }
            else {
                return Color(0.0);         // black
            }
        }
        else {
            // reflection
            Vec3d intersectionPosition(m_origin + (m_direction * tMin));
            Vec3d normal(closest->getNormal(intersectionPosition));
            Ray reflectedRay(intersectionPosition,
                             m_direction.getReflectedAt(normal).getNormalized(),
                             m_depth+1,*m_objList,*m_lightList);

            // calculate lighting
            for (int j=0; j<lightListSize; j++) {

                // where is the lightsource ?
                Ray rayoflight(intersectionPosition, m_lightListCuda[j].direction(), 0, *m_objList, *m_lightList);
                bool something_intersected = false;

                // where are the objects ?
                for (int k=0; k<objListSize; k++) {

                    double t = m_objListCuda[k].intersect(rayoflight);
                    if (t > 0.0) {
                        something_intersected = true;
                        break;
                    }

                } // for all obj

                // is it visible ?
                if (! something_intersected)
                    currentColor += shadedColor(&m_lightListCuda[j], reflectedRay, normal, closest);

            } // for all lights

            // could be right...
            currentColor *= closest->mirror();
        }
   }
   return Color(currentColor);
}

// Description:
// Determine color contribution of a lightsource
inline const Color __device__ __host__
Ray::shadedColor(LightObject *light, const Ray &reflectedRay, const Vec3d &normal, GeoObject *obj) const
{
   double ldot = light->direction() | normal;
   Color reflectedColor = Color(0.0);

   // lambertian reflection model
   if (ldot > 0.0)
      reflectedColor += obj->reflectance() * (light->color() * ldot);

   // updated with ambient lightning as in:
   // [GENERALISED AMBIENT REFLECTION MODELS FOR LAMBERTIAN AND PHONG SURFACES, Xiaozheng Zhang and Yongsheng Gao]
//   reflectedColor += obj->ambient() * g_sceneCuda.ambience;

   // specular part
   double spec = reflectedRay.direction() | light->direction();
   if (spec > 0.0) {
      spec = obj->specular() * pow(spec, obj->specularExp());
      reflectedColor += light->color() * spec;
   }

   return Color(reflectedColor);
}

inline Vec3d __device__ __host__
Ray::origin() const
{
   return Vec3d(m_origin);
}

inline Vec3d __device__ __host__
Ray::direction() const
{
   return Vec3d(m_direction);
}

