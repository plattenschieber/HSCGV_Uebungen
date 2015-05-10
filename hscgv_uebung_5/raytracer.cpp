/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "raytracer.h"

// Description:
// the connection to the parser
extern int openSceneFile(const char *s);
extern int closeSceneFile(void);
extern int inputparse(void);
extern int cleanUp(void);

//! get scene description from the parser
/*!
  Get the scene description from the parser.
  \param inputfile Filename of the scene
 */
void
ReadScene(const char *inputfile)
{
   openSceneFile(inputfile);
   inputparse();
   closeSceneFile();
}

// scene storage
//! the list of geometric objects in the scene
extern std::vector<GeoObject*>           g_objectList;
//! the list of light sources in the scene
extern std::vector<LightObject*>         g_lightList;
//! the list of surface properties of the objects
extern std::vector<GeoObjectProperties*> g_propList;

Raytracer::Raytracer(char *filename, char *outfilename, bool antialiasing):filename(filename),
    outfilename(outfilename),
    antialiasing(antialiasing)
{
   // parse the input file
   ReadScene(m_filename);
}

void
Raytracer::start() {
   unsigned int maxColVal   = 255;

   unsigned int cCol[3];

   // prepare byte stream for rgb data
   unsigned char * data;
   data = (unsigned char*)malloc( sizeof(unsigned char) * g_scene.picture.Xresolution * g_scene.picture.Yresolution * 3 );

   // TODO take view parameters from file into account and remove hardcoded values
   // setup viewport, its origin is bottom left

   // setup camera coordsys
   Vec3d eye_dir = (g_scene.view.lookat - g_scene.view.eyepoint).getNormalized();
   Vec3d eye_right = (eye_dir^g_scene.view.up).getNormalized();
   Vec3d eye_up = eye_dir^eye_right*-1;

    // calculatehe dimensions of the viewport using the scene's camera
    float height = 2 * tan(M_PI/180 * .5 * g_scene.view.fovy);
    float width = height * g_scene.view.aspect;

    // compute delta steps in each direction
    Vec3d deltaX = eye_right * (width / g_scene.picture.Xresolution);
    Vec3d deltaY = eye_up * (height / g_scene.picture.Yresolution);

    // this should be bottom left
    Vec3d bottomLeft = g_scene.view.eyepoint + eye_dir - deltaX*g_scene.picture.Xresolution/2 - deltaY*g_scene.picture.Yresolution/2;

   // normal ray tracing: the color of the center of a pixel is computed
   for (unsigned int sy=g_scene.picture.Yresolution ; sy > 0 ; --sy) {
      for (unsigned int sx=0 ; sx < g_scene.picture.Xresolution ; ++sx) {
         // the center of the pixel we are looking at right now
         Vec3d point = bottomLeft + deltaX*sx + deltaY*sy + deltaX/2 + deltaY/2;

         // the direction of our look
         Vec3d dir = point - g_scene.view.eyepoint;

         // create ray from view.eyepoint to view.lookat
         Ray theRay(g_scene.view.eyepoint,dir.getNormalized(),0,g_objectList,g_lightList);

         // compute the color
         Color col = theRay.shade();

         // in case we are using antialiasing, calculate the color of this pixel by averaging
         if (m_antialiasing) {
               // scale the midpoint color since we are going to use 5 points to average our color
               col *= 0.2;

               // besides taking shooting a ray through the midpoint 'o', we calculate
               // the pixels color by shooting 4 more rays  through the points 'x'
               // and averaging their values
               //  -------
               // | x   x |
               // |   o   |
               // | x   x |
               //  --------
               for(float dx = -1/4.; dx <= 1/4.; dx+=1/2.)
               {
                 for(float dy = -1/4.; dy <= 1/4.; dy+=1/2.)
                 {
                   Vec3d superSamplePoint = point + deltaX*dx + deltaY*dy;
                   Vec3d superSampleDir = superSamplePoint - g_scene.view.eyepoint;

                   // create ray from view.eyepoint to view.lookat
                   Ray theRay(g_scene.view.eyepoint,superSampleDir.getNormalized(),0,g_objectList,g_lightList);
                   col += theRay.shade()*0.2;//color, recursive_ray_trace(eye, ray, 0));
                 }
               }
         }

         // clamp the computed color value to 0...maxColVal
         for (unsigned int cc=0; cc<3; cc++) {
            if(col[cc] < 0.0)
               cCol[cc] = 0;
            else if(col[cc] > 1.0)
               cCol[cc] = maxColVal;
            else
               cCol[cc] = (unsigned int)(maxColVal * col[cc]);
         }

         int index = 3*((sy-1) * g_scene.picture.Xresolution + sx);
         data[index + 0] = '\0';
         data[index + 1] = '\0';
         data[index + 2] = '\0';
//         data[index + 0] = (char)cCol[0];
//         data[index + 1] = (char)cCol[1];
//         data[index + 2] = (char)cCol[2];
      } // foreach x
   } // foreach y

   // clean up
   cleanUp();

}
