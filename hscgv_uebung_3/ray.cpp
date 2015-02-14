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
Ray::Ray( )
: m_origin()
, m_direction()
, m_depth(0)
, m_objList(NULL)
, m_lightList(NULL)
{
   ERR("don't use uninitialized rays !");
}

// Description:
// Copy-Constructor.
Ray::Ray( const Ray &r )
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
Ray::Ray(const Vec3d &o, const Vec3d &d, unsigned int i, std::vector<GeoObject*> &ol, std::vector<LightObject*> &ll)
{
   // copy it ! initialization is not enough !
   m_origin    = o;
   m_direction = d;
   m_depth     = i;
   m_objList   = &ol;
   m_lightList = &ll;
}

// Description:
// Destructor.
Ray::~Ray( )
{
}

// Description:
// Determine color of this ray by tracing through the scene
const Color
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
// Determine color contribution of a lightsource
const Color
Ray::shadedColor(LightObject *light, const Ray &reflectedRay, const Vec3d &normal, GeoObject *obj) const
{
   double ldot = light->direction() | normal;
   Color reflectedColor = Color(0.0);

   // lambertian reflection model
   // updated with ambient lightning as in:
   // [GENERALISED AMBIENT REFLECTION MODELS FOR LAMBERTIAN AND PHONG SURFACES, Xiaozheng Zhang and Yongsheng Gao]
   if (ldot > 0.0)
      reflectedColor += obj->reflectance() * (light->color() * ldot) + obj->ambient() * g_scene.ambience;

   // specular part
   double spec = reflectedRay.direction() | light->direction();
   if (spec > 0.0) {
      spec = spec * spec;
      spec = spec * spec;
      spec = spec * spec;
      spec *= obj->specular();
      reflectedColor += light->color() * spec;
   }

   return Color(reflectedColor);
}

Vec3d
Ray::origin() const
{
   return Vec3d(m_origin);
}

Vec3d
Ray::direction() const
{
   return Vec3d(m_direction);
}
