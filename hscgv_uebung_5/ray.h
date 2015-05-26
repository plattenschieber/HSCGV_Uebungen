/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: ray.h
 *   Declaration of rays traversing the scene
 */

#ifndef RAY_HH
#define RAY_HH

#include <vector>
#include "vector.h"
#include "geoobject.h"
#include "lightobject.h"

//! A ray
/*! This type of object describes a ray, half a line starting
  at its origin, continuing in only one direction
 */
class Ray
{
   public:
      // CONSTRUCTORS
      __host__ __device__ Ray();
      __host__ __device__ Ray(const Ray &r);
      __host__ Ray(const Vec3d &o, const Vec3d &d, unsigned int i, std::vector<GeoObject*> &ol, std::vector<LightObject*> &ll);
      __device__ Ray(const Vec3d o, const Vec3d d, unsigned int i);
      __host__ __device__ ~Ray();

      const Color __host__ shade() const;
      const Color __device__ shade(Ray *thisRay, Vec3d d_origin, Vec3d d_direction, GeoObject *d_objList, int objListSize, LightObject *d_lightList, int lightListSize, Color background) ;

      // access methods
      Vec3d __host__ __device__ origin() const;
      Vec3d __host__ __device__ direction() const;

   private:
      Vec3d        m_origin;
      Vec3d        m_direction;
      unsigned int m_depth;

      std::vector<GeoObject*>   *m_objList;
      std::vector<LightObject*> *m_lightList;
};
#include "ray.inl"

#endif /* RAY_HH */
