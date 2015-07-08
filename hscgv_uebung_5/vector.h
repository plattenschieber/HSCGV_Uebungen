/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 3 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: vector.h
 *   A template for 3-dimensional vectors
 */

#ifndef VECTOR_HH
#define VECTOR_HH

#include <iostream>
#include <cmath>

#include "types.h"

//! should we initialize with zero ?
/*!
  When a new Vector is created without any value argument, should the values 
  initially be set to zero ?
 */
#define VECTOR_INIT_ZERO true

template<class Scalar> class Vector;

template<class Scalar>
std::ostream&
operator<<( std::ostream& os, const Vector<Scalar>& i );

template<class Scalar>
std::istream&
operator>>( std::istream& is, Vector<Scalar>& i );

//! The basic Vector class
/*!
  This class contains the basic mathematical tool: a three-dimensional vector.
  It can be used right out-of-the-box and has all algebraic methods already
  implemented. You will need to make changes here only in very rare cases.
 */
template<class Scalar>
class Vector
{
    public:
        typedef Scalar value_type;

   public:
      //! Constructor
      __host__ __device__ Vector(void );

      //! Copy-Constructor
      __host__ __device__ Vector(const Vector<Scalar> &v );

      //! construct a Vector from one Scalar
      __host__ __device__ Vector(Scalar);

      //! construct a Vector from three Scalars
      __host__ __device__ Vector(Scalar, Scalar, Scalar);

      //! Assignment operator
      __host__ __device__ const Vector<Scalar>& operator=  (const Vector<Scalar>& v);

      //! Assignment operator
       __device__ __host__ const Vector<Scalar>& operator=  (Scalar s);

      //! Assign and add operator
       __device__ __host__ const Vector<Scalar>& operator+= (const Vector<Scalar>& v);

      //! Assign and add operator
       __device__ __host__ const Vector<Scalar>& operator+= (Scalar s);

      //! Assign and sub operator
       __device__ __host__ const Vector<Scalar>& operator-= (const Vector<Scalar>& v);

      //! Assign and sub operator
       __device__ __host__ const Vector<Scalar>& operator-= (Scalar s);

      //! Assign and mult operator
       __device__ __host__ const Vector<Scalar>& operator*= (const Vector<Scalar>& v);

      //! Assign and mult operator
       __device__ __host__ const Vector<Scalar>& operator*= (Scalar s);

      //! Assign and div operator
       __device__ __host__ const Vector<Scalar>& operator/= (const Vector<Scalar>& v);

      //! Assign and div operator
       __device__ __host__ const Vector<Scalar>& operator/= (Scalar s);

      //! output operator
          __device__ __host__ friend std::ostream& operator<< <>( std::ostream& os, const Vector<Scalar>& inst );

      //! input operator
         friend std::istream& operator>> <>( std::istream& is, Vector<Scalar>& inst );

      //! unary operator
      __host__ __device__ Vector<Scalar> operator- () const;

      //! binary operator add
      __host__ __device__ Vector<Scalar> operator+ (const Vector<Scalar>&) const;

      //! binary operator add
      __host__ __device__ Vector<Scalar> operator+ (Scalar) const;

      //! binary operator sub
      __host__ __device__ Vector<Scalar> operator- (const Vector<Scalar>&) const;

      //! binary operator sub
      __host__ __device__ Vector<Scalar> operator- (Scalar) const;

      //! binary operator mult
      __host__ __device__ Vector<Scalar> operator* (const Vector<Scalar>&) const;

      //! binary operator mult
      __host__ __device__ Vector<Scalar> operator* (Scalar) const;

      //! binary operator div
      __host__ __device__ Vector<Scalar> operator/ (const Vector<Scalar>&) const;

      //! binary operator div
      __host__ __device__ Vector<Scalar> operator/ (Scalar) const;

      //! equality test operator
      __host__ __device__ bool operator== (const Vector<Scalar>&) const;

      //! inequality test operator
      __host__ __device__ bool operator!= (const Vector<Scalar>&) const;

      //! scalar product
      __host__ __device__ Scalar operator|( const Vector<Scalar> &v ) const;

      //! norm
      __host__ __device__ Scalar getNorm() const;

      //! normalize
      __host__ __device__ Vector<Scalar> getNormalized() const;

      //! normalize and norm
      __host__ __device__ Scalar normalize();

      //! Vector product
      __host__ __device__ Vector<Scalar> operator^( const Vector<Scalar> &v ) const;

      //! Projection normal to a vector
      __host__ __device__ Vector<Scalar>	  getOrthogonalVector() const;

      //! Project into a plane
      __host__ __device__ const Vector<Scalar>& projectNormalTo(const Vector<Scalar> &v);

      //! Reflect at normal
      __host__ __device__ Vector<Scalar>	  getReflectedAt(const Vector<Scalar>& n) const;

      //! Refract at normal
      __host__ __device__ Vector<Scalar>	  getRefractedAt(const Vector<Scalar>& n,
            double index,
            bool& totalReflection) const;

      //! minimize
      __host__ __device__ const Vector<Scalar> &minimize(const Vector<Scalar> &);

      //! maximize
      __host__ __device__ const Vector<Scalar> &maximize(const Vector<Scalar> &);

      //! access operator
      __host__ __device__ Scalar& operator[](unsigned int i);

      //! access operator
      __host__ __device__ const Scalar& operator[](unsigned int i) const;

      //! Get minimal vector length
      __host__ __device__ static Scalar getEpsilon( );

   protected:

   private:
      Scalar m_value[3];  //!< Storage of Vector values
};


// Description:
/*!
  Outputs the object in human readable form using the format
  [x,y,z]
 */
template<class Scalar>
std::ostream&
operator<<( std::ostream& os, const Vector<Scalar>& i )
{
   os << '[' << i.m_value[0] << ", " << i.m_value[1] << ", " << i.m_value[2] << ']';
   return os;
}

// Description:
/*!
  Reads the contents of the object from a stream using the same format
  as the output operator.
 */
template<class Scalar>
std::istream&
operator>>( std::istream& is, Vector<Scalar>& i )
{
   char c;
   char dummy[3];

   is >> c >> i.m_value[0] >> dummy >> i.m_value[1] >> dummy >> i.m_value[2] >> c;
   return is;
}

//! a 3D Vector with double precision
/*!
  The standard Vectors we're working with have three dimensions of double
  precision values. This is necessary to ensure accuracy of intersection
  computation for example.
 */
typedef Vector<double> Vec3d; 

//! a 3D Vector with single precision
/*!
  For performance of memory saving reasons, you can experiment with
  single precision values. In this case however, you have to pay more
  attention to the correctness of the results.
 */
typedef Vector<float>  Vec3f; 

//! a 3D integer Vector
/*!
  For special cases, here is a Vector with integer values. Use them to
  lookup something, etc.
 */
typedef Vector<int>    Vec3i; 

//! Color uses double precision values
/*!
  Color is defined to consist of three values: one for red, one for green and
  one for the blue channel. For most applications, this partition of the
  frequency spectrum has proven to be sufficient.
 */
typedef Vector<double> Color; 



/*
 * GCC needs templates to be defined in the header
 */

//! the minimal Vector length
/*!
  In order to be able to discriminate floating point values near zero, and
  to be sure not to fail a comparision because of roundoff errors, use this
  m_value as a threshold.
 */
#define VECTOR_EPSILON (1e-6)

// Description:
/*!
  Constructor.
 */
template<class Scalar> __device__ __host__
Vector<Scalar>::Vector( void )
{
#ifdef VECTOR_INIT_ZERO
   m_value[0] = m_value[1] = m_value[2] = 0;
#endif
}


// Description:
/*!
  Copy-Constructor.
 */
template<class Scalar> __device__ __host__
Vector<Scalar>::Vector( const Vector<Scalar> &v )
{
   m_value[0] = v.m_value[0];
   m_value[1] = v.m_value[1];
   m_value[2] = v.m_value[2];
}


// Description:
/*!
  Constructor for a Vector from a single Scalar. All components of
  the Vector get the same m_value.
  \param s The m_value to set
  \return The new Vector
 */
template<class Scalar> __device__ __host__
Vector<Scalar>::Vector(Scalar s )
{
   m_value[0]= s;
   m_value[1]= s;
   m_value[2]= s;
}

// Description:
/*!
  Constructor for a Vector from three Scalars.
  \param s1 The value for the first Vector component
  \param s2 The value for the second Vector component
  \param s3 The value for the third Vector component
  \return The new Vector
 */
template<class Scalar> __device__ __host__
Vector<Scalar>::Vector(Scalar s1, Scalar s2, Scalar s3)
{
   m_value[0]= s1;
   m_value[1]= s2;
   m_value[2]= s3;
}

// Description:
/*!
  Compute the vector product of two 3D vectors
  \param v Second vector to compute the product with
  \return A new Vector with the product values
 */
template<class Scalar> __device__ __host__
Vector<Scalar> 
Vector<Scalar>::operator^( const Vector<Scalar> &v ) const
{
   return Vector<Scalar>(m_value[1]*v.m_value[2] - m_value[2]*v.m_value[1],
         m_value[2]*v.m_value[0] - m_value[0]*v.m_value[2],
         m_value[0]*v.m_value[1] - m_value[1]*v.m_value[0]);
}

// Description:
/*!
  Copy a Vector componentwise.
  \param v Vector with values to be copied
  \return Reference to self
 */
template<class Scalar> __device__ __host__
const Vector<Scalar>&
Vector<Scalar>::operator=( const Vector<Scalar> &v )
{
   m_value[0] = v.m_value[0];
   m_value[1] = v.m_value[1];
   m_value[2] = v.m_value[2];  
   return *this;
}

// Description:
/*!
  Copy a Scalar to each component.
  \param s The value to copy
  \return Reference to self
 */
template<class Scalar> __device__ __host__
const Vector<Scalar>&
Vector<Scalar>::operator=(Scalar s)
{
   m_value[0] = s;
   m_value[1] = s;
   m_value[2] = s;  
   return *this;
}

// Description:
/*!
  Add another Vector componentwise.
  \param v Vector with values to be added
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator+=( const Vector<Scalar> &v )
{
   m_value[0] += v.m_value[0];
   m_value[1] += v.m_value[1];
   m_value[2] += v.m_value[2];  
   return *this;
}

// Description:
/*!
  Add a Scalar value to each component.
  \param s value to add
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator+=(Scalar s)
{
   m_value[0] += s;
   m_value[1] += s;
   m_value[2] += s;  
   return *this;
}

// Description:
/*!
  Subtract another Vector componentwise.
  \param v Vector of values to subtract
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator-=( const Vector<Scalar> &v )
{
   m_value[0] -= v.m_value[0];
   m_value[1] -= v.m_value[1];
   m_value[2] -= v.m_value[2];  
   return *this;
}

// Description:
/*!
  Subtract a Scalar value from each component.
  \param s value to subtract
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator-=(Scalar s)
{
   m_value[0]-= s;
   m_value[1]-= s;
   m_value[2]-= s;  
   return *this;
}

// Description:
/*!
  Multiply with another Vector componentwise.
  \param v Vector of values to multiply with
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator*=( const Vector<Scalar> &v )
{
   m_value[0] *= v.m_value[0];
   m_value[1] *= v.m_value[1];
   m_value[2] *= v.m_value[2];  
   return *this;
}

// Description:
/*!
  Multiply each component with a Scalar value.
  \param s value to multiply with
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator*=(Scalar s)
{
   m_value[0] *= s;
   m_value[1] *= s;
   m_value[2] *= s;  
   return *this;
}

// Description:
/*!
  Divide by another Vector componentwise.
  \param v Vector of values to divide by
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator/=( const Vector<Scalar> &v )
{
   m_value[0] /= v.m_value[0];
   m_value[1] /= v.m_value[1];
   m_value[2] /= v.m_value[2];  
   return *this;
}

// Description:
/*!
  Divide each component by a Scalar value.
  \param s value to divide by
  \return Reference to self
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::operator/=(Scalar s)
{
   m_value[0] /= s;
   m_value[1] /= s;
   m_value[2] /= s;
   return *this;
}

// ===============
// unary operators
// ===============

// Description:
/*!
  Build componentwise the negative this Vector.
  \return The new (negative) Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator-() const
{
   return Vector<Scalar>(-m_value[0], -m_value[1], -m_value[2]);
}

// ================
// binary operators
// ================

// Description:
/*!
  Build a Vector with another Vector added componentwise.
  \param v The second Vector to add
  \return The sum Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator+( const Vector<Scalar> &v ) const
{
   return Vector<Scalar>(m_value[0]+v.m_value[0],
         m_value[1]+v.m_value[1],
         m_value[2]+v.m_value[2]);
}

// Description:
/*!
  Build a Vector with a Scalar m_value added to each component.
  \param s The Scalar m_value to add
  \return The sum Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator+(Scalar s) const
{
   return Vector<Scalar>(m_value[0]+s,
         m_value[1]+s,
         m_value[2]+s);
}

// Description:
/*!
  Build a Vector with another Vector subtracted componentwise.
  \param v The second Vector to subtract
  \return The difference Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator-( const Vector<Scalar> &v ) const
{
   return Vector<Scalar>(m_value[0]-v.m_value[0],
         m_value[1]-v.m_value[1],
         m_value[2]-v.m_value[2]);
}

// Description:
/*!
  Build a Vector with a Scalar value subtracted componentwise.
  \param s The Scalar value to subtract
  \return The difference Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator-(Scalar s ) const
{
   return Vector<Scalar>(m_value[0]-s,
         m_value[1]-s,
         m_value[2]-s);
}


// Description:
/*!
  Build a Vector with another Vector multiplied by componentwise.
  \param v The second Vector to muliply with
  \return The product Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator*( const Vector<Scalar>& v) const
{
   return Vector<Scalar>(m_value[0]*v.m_value[0],
         m_value[1]*v.m_value[1],
         m_value[2]*v.m_value[2]);
}

// Description:
/*!
  Build a Vector with a Scalar value multiplied to each component.
  \param s The Scalar value to multiply with
  \return The product Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator*(Scalar s) const
{
   return Vector<Scalar>(m_value[0]*s, m_value[1]*s, m_value[2]*s);
}

// Description:
/*!
  Build a Vector divided componentwise by another Vector.
  \param v The second Vector to divide by
  \return The ratio Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator/(const Vector<Scalar>& v) const
{
   return Vector<Scalar>(m_value[0]/v.m_value[0],
         m_value[1]/v.m_value[1],
         m_value[2]/v.m_value[2]);
}


// Description:
/*!
  Build a Vector divided componentwise by a Scalar value.
  \param s The Scalar value to divide by
  \return The ratio Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::operator/(Scalar s) const
{
   return Vector<Scalar>(m_value[0]/s,
         m_value[1]/s,
         m_value[2]/s);
}


// Description:
/*!
  Test two Vectors for equality based on the equality of their values within a small threshold.
  \param c The second Vector to compare
  \return TRUE if both are equal
  \sa getEpsilon()
 */
template<class Scalar>
bool
Vector<Scalar>::operator== (const Vector<Scalar>& c) const
{
   return (ABS(m_value[0]-c.m_value[0]) + 
         ABS(m_value[1]-c.m_value[1]) + 
         ABS(m_value[2]-c.m_value[2]) < VECTOR_EPSILON);
}

// Description:
/*!
  Test two Vectors for inequality based on the inequality of their
  values within a small threshold.
  \param c The second Vector to compare
  \return FALSE if both are equal
  \sa getEpsilon()
 */
template<class Scalar>
bool
Vector<Scalar>::operator!= (const Vector<Scalar>& c) const
{
   return (!(ABS(m_value[0]-c.m_value[0]) + 
            ABS(m_value[1]-c.m_value[1]) + 
            ABS(m_value[2]-c.m_value[2]) < VECTOR_EPSILON));
}




// Description:
/*!
  Get a particular component of the vector.
  \param i Number of Scalar to get
  \return Reference to the component
 */
template<class Scalar>
Scalar&
Vector<Scalar>::operator[]( unsigned int i )
{
   return m_value[i];
}

// Description:
/*!
  Get a particular component of a constant vector.
  \param i Number of Scalar to get
  \return Reference to the component
 */
template<class Scalar>
const Scalar&
Vector<Scalar>::operator[]( unsigned int i ) const
{
   return m_value[i];
}



// ================
// scalar
// ================

// Description:
/*!
  Compute the scalar product with another Vector.
  \param v The second Vector to work with
  \return The value of the scalar product
 */
template<class Scalar>
Scalar
Vector<Scalar>::operator|(const Vector<Scalar> &v ) const
{
   return m_value[0]*v.m_value[0] + m_value[1]*v.m_value[1] + m_value[2]*v.m_value[2];
}


// ================
// norm
// ================

// Description:
/*!
  Compute the length (norm) of the Vector.
  \return The value of the norm
 */
template<class Scalar>
Scalar
Vector<Scalar>::getNorm() const
{
   Scalar l = m_value[0]*m_value[0] + m_value[1]*m_value[1] + m_value[2]*m_value[2];
   return sqrt(l);
}

// Description:
/*!
  Compute a normalized Vector based on this Vector.
  \return The new normalized Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::getNormalized() const
{
   Scalar l = m_value[0]*m_value[0] + m_value[1]*m_value[1] + m_value[2]*m_value[2];
   if (fabs(l-1.) < VECTOR_EPSILON*VECTOR_EPSILON)
      return *this;
   else if (l > VECTOR_EPSILON*VECTOR_EPSILON)
   {
      double fac = 1./sqrt(l);
      return Vector<Scalar>(m_value[0]*fac, m_value[1]*fac, m_value[2]*fac);
   }
   else
      return Vector<Scalar>((Scalar)0);
}

// Description:
/*!
  Compute the norm of the Vector and normalize it.
  \return The value of the norm
 */
template<class Scalar>
Scalar
Vector<Scalar>::normalize()
{
   double norm;
   Scalar l = m_value[0]*m_value[0] + m_value[1]*m_value[1] + m_value[2]*m_value[2];  
   if (l < VECTOR_EPSILON*VECTOR_EPSILON)
   {
      m_value[0]= m_value[1]= m_value[2]= 0;
      norm = 0.;
   }
   else if (fabs(l-1.) > VECTOR_EPSILON*VECTOR_EPSILON)
   {
      norm = sqrt(l);
      double fac = 1./norm;
      m_value[0] *= fac;
      m_value[1] *= fac;
      m_value[2] *= fac;
   }
   else
      norm = 1.;

   return (Scalar)norm;
}

// Description:
/*!
  Compute a Vector that is orthonormal to self. Nothing else can be assumed
  for the direction of the new Vector.
  \return The orthonormal Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::getOrthogonalVector() const
{
   //! Determine the  component with max. absolute value
   int max= (fabs(m_value[0]) > fabs(m_value[1])) ? 0 : 1;
   max= (fabs(m_value[max]) > fabs(m_value[2])) ? max : 2;

   /*!
     Choose another axis than the one with max. component and project
     orthogonal to self
    */
   Vector<Scalar> vec(0.0);
   vec[(max+1)%3]= 1;
   vec.normalize();
   vec.projectNormalTo(this->getNormalized());
   return vec;
}

// Description:
/*!
  Projects the vector into a plane normal to the given vector, which must
  have unit length. Self is modified.
  \param v The plane normal
  \return The projected Vector
 */
template<class Scalar>
const Vector<Scalar>&
Vector<Scalar>::projectNormalTo(const Vector<Scalar> &v)
{
   Scalar sprod = (*this|v);
   m_value[0]= m_value[0] - v.m_value[0] * sprod;
   m_value[1]= m_value[1] - v.m_value[1] * sprod;
   m_value[2]= m_value[2] - v.m_value[2] * sprod;  
   return *this;
}


// Description:
/*!
  Compute a Vector, that is self (as an incoming
  Vector) reflected at a surface with a distinct normal vector. Note
  that the normal is reversed, if the scalar product with it is positive.
  \param n The surface normal
  \return The new reflected Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::getReflectedAt(const Vector<Scalar>& n) const
{
   Vector<Scalar> nn= ((*this | n) > 0) ? -n : n;
   return *this - nn * (2 * (nn|*this));
}

// Description:
/*!
  Compute a Vector, that is self refracted
  (as an incoming vector) at a surface with a distinct normal vector and a
  index of refraction. The refraction ratio is defined in the sense, that the
  numer is the first material along the self vector. In case of total internal
  reflection, a flag is set and self is returned unmodified.
  \param n The surface normal
  \param index The index of refraction
  \param totalReflection TRUE if total internal reflection occurs
  \return The modified (refracted) Vector
 */
template<class Scalar>
Vector<Scalar>
Vector<Scalar>::getRefractedAt(const Vector<Scalar>& n, double index,
      bool& totalReflection ) const
{
   // Reorient n
   Vector<Scalar> nn= ((*this | n) > 0) ? -n : n;
   Vector<Scalar> tmp= (*this - nn * (nn|*this)) * index;
   double length= 1 - (tmp|tmp);
   totalReflection= (length < 0) ? true : false;
   return (length < 0) ? *this : nn * -sqrt(length) + tmp;
}


// Description:
/*!
  Minimize the Vector, i.e. set each entry of the Vector to the minimum
  of both values.
  \param pnt The second Vector to compare with
  \return Reference to the modified self
 */
template<class Scalar>
const Vector<Scalar> &
Vector<Scalar>::minimize(const Vector<Scalar> &pnt)
{
   for (unsigned int i = 0; i < 3; i++)
      m_value[i] = MIN(m_value[i],pnt[i]);
   return *this;
}


// Description:
/*!
  Maximize the Vector, i.e. set each entry of the Vector to the maximum
  of both values.
  \param pnt The second Vector to compare with
  \return Reference to the modified self
 */
template<class Scalar>
const Vector<Scalar> &
Vector<Scalar>::maximize(const Vector<Scalar> &pnt)
{
   for (unsigned int i = 0; i < 3; i++)
      m_value[i] = MAX(m_value[i],pnt[i]);
   return *this;
}


// ================
// VECTOR_EPSILON handle
// ================

// Description:
/*!
  Get minimal vector length value that can be discriminated.
  \return The minimal value
  \sa VECTOR_EPSILON
 */
template<class Scalar>
Scalar
Vector<Scalar>::getEpsilon()
{
   return VECTOR_EPSILON;
}



#endif /* VECTOR_HH */
