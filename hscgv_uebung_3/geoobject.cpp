/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "geoobject.h"

/* member access functions for GeoObjectProperties */

Vec3d
GeoObjectProperties::reflectance() const
{
   return Vec3d(m_reflectance);
}

double
GeoObjectProperties::specular() const
{
   return m_specular;
}

double
GeoObjectProperties::mirror() const
{
   return m_mirror;
}

/* access functions for the properties of a GeoObject */

Vec3d
GeoObject::reflectance() const
{
   if (m_properties)
      return m_properties->reflectance();
   else
      std::cerr<<"WARNING: properties not set"<<std::endl;
   return Vec3d(0.0);
}

double
GeoObject::specular() const
{
   if (m_properties)
      return m_properties->specular();
   else
      std::cerr<<"WARNING: properties not set"<<std::endl;
   return 0.0;
}

double
GeoObject::mirror() const
{
   if (m_properties)
      return m_properties->mirror();
   else
      std::cerr<<"WARNING: properties not set"<<std::endl;
   return 0.0;
}

void
GeoObject::setProperties(GeoObjectProperties *p)
{
   m_properties = p;
}

