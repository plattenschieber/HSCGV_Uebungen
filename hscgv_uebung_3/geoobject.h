/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: geoobject.h
 *   Declaration of abstract geometric objects and their properties
 */

#ifndef GEOOBJECT_HH
#define GEOOBJECT_HH

#include "vector.h"
class Ray;

//! the surface properties of a GeoObject
/*! This class stores the surface properties of a GeoObject,
  such as its ambient and diffuse color, the parameters
  describing specular reflection and its ability to mirror light.
 */
class GeoObjectProperties
{
   public:
      // CONSTRUCTORS
      GeoObjectProperties() {};
      GeoObjectProperties(const Vec3d& a,const Vec3d& r, double s, int e, double m) {
         m_ambient     = a;
         m_reflectance = r;
         m_specular    = s;
         m_specularExp = e;
         m_mirror      = m;
      };
      ~GeoObjectProperties() {};

      // access methods
      Color  ambient() const;
      Vec3d  reflectance() const;
      double specular() const;
      int    specularExp() const;
      double mirror() const;

   protected:
      Color  m_ambient;
      Vec3d  m_reflectance;
      double m_specular;
      int    m_specularExp;
      double m_mirror;
};

//! A geometric object
/*! This class abstracts a general geometric object. This includes its
 *  its surface properties and the possibility to calculate intersection
 *  points with rays and surface normals */
class GeoObject
{
   public:
      // CONSTRUCTORS
      GeoObject() {};
      virtual ~GeoObject() {};
      virtual Vec3d  getNormal(const Vec3d &v) const = 0;
      virtual double intersect(const Ray &r) const = 0;

      // access methods
      virtual Color  ambient() const;
      virtual Vec3d  reflectance() const;
      virtual double specular() const;
      virtual int    specularExp() const;

      virtual double mirror() const;

      // config method
      virtual void setProperties(GeoObjectProperties *p);

   protected:
      GeoObjectProperties *m_properties;
};

#endif /* GEOOBJECT_HH */
