/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: lightobject.h
 *   Declaration of light sources
 */
#ifndef LIGHTOBJECT_HH
#define LIGHTOBJECT_HH

#include "vector.h"


//! A light source
/*! This class describes light sources positioned at infinity.
 */
class LightObject
{
   public:
      // CONSTRUCTORS
      LightObject();
      LightObject(const Vec3d& dir, const Color& col);
      virtual ~LightObject();

      // access methods
      Color __device__ __host__ color() const;
      Vec3d __device__ __host__ direction() const;

   protected:
      typedef struct {
         Color  color;
         Vec3d  direction;
      } LightObjectProperties;

   private:
      LightObjectProperties *m_properties;
};

#include "lightobject.inl"

#endif /* LIGHTOBJECT_HH */
