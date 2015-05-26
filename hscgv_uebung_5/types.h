/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: types.h
 *   Miscellaneous macros and templates
 */

#ifndef TYPES_HH
#define TYPES_HH

#include <iostream>
#include <cstdio>

// ignore all compiler errors due to mixing up of cuda host and device code
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

//! error message macro
/*!
  Use this macro to generate informative output in case the program detects
  an error of any kind. Normally your program should exit cleanly after
  this output. It is bad code to detect a error condition and not being able
  to exit cleanly.
 */
#define ERR(A) {std::cerr << "ERROR in " << __FILE__ << " at line " << __LINE__ << ": " << A << std::endl;}

#if defined(TRACE) || defined(VERBOSE)
//! user message macro
/*!
  Use this macro to generate user notification messages to indicate the
  program progress. Because the code is only inserted, if the macro 
  VERBOSE is defined, it allows to build more silent program versions.
  You should only indicate error conditions with this macro, that can be
  corrected automatically by the program.
  \sa ERR(A)
 */
#define VOUT(A) {std::cerr << A << std::endl;}
#else
#define VOUT(A)
#endif

#ifdef TRACE
//! program tracing macro
/*!
  Use this macro to generate detailed debug messages. The code is only
  inserted if the macro TRACE is defined. When compiling optimized program
  versions, these messages should be suppressed to minimize system calls.
  \sa ERR(A) VOUT(A)
 */
#define TOUT(A) {fprintf(stderr,"%12s(%4d)#> ",__FILE__,__LINE__); fprintf A;}
#else
#define TOUT(A) 
#endif


//! minimum
/*!
  Determine the minimum of two items. Note that this is a template, so
  the appropriate operator< is used.
  \param a The first item
  \param b The second item
  \return The smaller value of both
 */
template < class T >
inline T
MIN( T a, T b )
{ return (a < b) ? a : b ; }

//! maximum
/*!
  Determine the maximum of two items. Note that this is a template, so
  the appropriate operator< is used.
  \param a The first item
  \param b The second item
  \return The bigger value of both
 */
template < class T >
inline T
MAX( T a, T b )
{ return (a < b) ? b : a ; }

//! absolute value
/*!
  Determine the absolute value of a item. Note that this is a template, so
  the appropriate unary operator- and operator< are used.
  \param a The item to check
  \return The absolute value
 */
template < class T >
inline T
ABS( T a )
{ return (0 < a) ? a : -a ; }

//! sign
/*!
  Determine the sign of the item. Note that this is a template, so the
  appropriate operator< is used.
  \param a The item to check
  \return 1 if item parameter value is positiv, -1 else
 */
template < class T >
inline T
SIGNUM( T a )
{ return (0 < a) ? 1 : -1 ; }

//! extended sign
/*!
  Determine the sign of the item. Additionally test for equality of zero.
  Note that this is a template, so the appropriate operators are used.
  \param a The item to check
  \return 1 if item parameter value is positiv, 0 if it's equal to zero, -1 else
 */
template < class T >
inline T
SIGNUM0( T a )
{ return (0 < a) ? 1 : ( a < 0 ? -1 : 0 ) ; }

//! round
/*!
  Convert floating point value to the nearest integer value.
  \param d Floating point value to convert
  \return Rounded integer
 */
inline int
ROUND(double d)
{ return int(d + 0.5); }


#endif /* TYPES_HH */
