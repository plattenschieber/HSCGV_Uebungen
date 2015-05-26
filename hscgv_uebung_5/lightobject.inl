/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "lightobject.h"

// Description:
// Constructor.
inline LightObject::LightObject()
: m_properties(NULL)
{
}

// Description:
// explicit parametrization
inline LightObject::LightObject(const Vec3d& dir, const Color& col)
{
   m_properties = new LightObjectProperties();
   m_properties->direction = dir.getNormalized();
   m_properties->color = col;
}

// Description:
// Destructor.
inline LightObject::~LightObject()
{
   delete m_properties;
}

inline __device__ __host__ Color
LightObject::color() const
{
   if (m_properties)
      return Color(m_properties->color);
//   else
//      std::cerr<<"WARNING: properties not set"<<std::endl;
   return Color(0.0);
}

inline __device__ __host__ Vec3d
LightObject::direction() const
{
   if (m_properties)
      return Vec3d(m_properties->direction);
//   else
//      std::cerr<<"WARNING: properties not set"<<std::endl;
   return Vec3d(0.0);
}

inline void __host__ __device__
LightObject::setProperties(LightObjectProperties *p)
{
   m_properties = p;
}

