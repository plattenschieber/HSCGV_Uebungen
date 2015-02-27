/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: param.h
 *   Data structures for global parameters
 */
#ifndef PARAM_HH
#define PARAM_HH

#include "vector.h"

//! Viewing parameters
/*! Structure collecting all viewing parameters
 */
struct View {
   //! the observer's location
   Vec3d eyepoint;
   //! the point focussed by the observer
   Vec3d lookat;
   //! vector specifying the upper side of the picture 
   Vec3d up;
   //! the angle for the field of view in up direction (specified in degrees)
   double fovy;
   //! the aspect ratio
   double aspect;
};

//! Picture parameters
/*! Structure collecting all picture parameters
 */
struct Picture {
   //! the picture width in pixels
   unsigned Xresolution;
   //! the picture height in pixels
   unsigned Yresolution;
   //! the background color of the picture
   Color background;
};

//! Scene parameters
/*! Structure collecting all the scene parameters
 */
struct Scene {
   //! the picture parameters
   Picture picture;
   //! the viewing parameters
   View view;
   //! the ambient light
   Color ambience;
};

extern Scene g_scene;


#endif /* PARAM_HH */
