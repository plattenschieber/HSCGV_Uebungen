/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "geoobject.h"

/* member access functions for GeoObjectProperties */

inline Color
GeoObjectProperties::ambient() const
{
   return m_ambient;
}

inline Vec3d
GeoObjectProperties::reflectance() const
{
   return m_reflectance;
}

inline double
GeoObjectProperties::specular() const
{
   return m_specular;
}

inline int
GeoObjectProperties::specularExp() const
{
   return m_specularExp;
}

inline double
GeoObjectProperties::mirror() const
{
   return m_mirror;
}

/* access functions for the properties of a GeoObject */

inline Color __host__
GeoObject::ambient() const
{
    if (m_properties)
        return m_properties->ambient();
    else
//        std::cerr<<"WARNING: properties not set"<<std::endl;
        return Vec3d(0.0);
}

inline Vec3d
GeoObject::reflectance() const
{
   if (m_properties)
      return m_properties->reflectance();
   else
//      std::cerr<<"WARNING: properties not set"<<std::endl;
       return Vec3d(0.0);
}

inline double
GeoObject::specular() const
{
   if (m_properties)
      return m_properties->specular();
   else
//      std::cerr<<"WARNING: properties not set"<<std::endl;
       return 0.0;
}

inline double
GeoObject::specularExp() const
{
   if (m_properties)
      return m_properties->specularExp();
   else
//      std::cerr<<"WARNING: properties not set"<<std::endl;
       return 0.0;
}

inline double
GeoObject::mirror() const
{
   if (m_properties)
      return m_properties->mirror();
   else
//      std::cerr<<"WARNING: properties not set"<<std::endl;
       return 0.0;
}

inline void
GeoObject::setProperties(GeoObjectProperties *p)
{
   m_properties = p;
}

