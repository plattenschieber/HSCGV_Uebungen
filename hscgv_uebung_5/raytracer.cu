/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "raytracer.h"
#include "omp.h"

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


Raytracer::Raytracer():
    m_isFileLoaded(false),
    m_filename(0),
    m_antialiasing(false) {}
Raytracer::Raytracer(bool antialiasing):
    m_isFileLoaded(false),
    m_filename(0),
    m_antialiasing(antialiasing) {}
Raytracer::Raytracer(const char *filename, bool antialiasing):
    m_filename(filename),
    m_antialiasing(antialiasing)
{
   // parse the input file
   ReadScene(m_filename);
   m_isFileLoaded = true;
}
Raytracer::~Raytracer() {
   // clean up
   cleanUp();
}

void
Raytracer::start(float *renderedScene, int xRes, int yRes) {
   // setup viewport, its origin is bottom left
   // setup camera coordsys
   Vec3d eye_dir = (g_scene.view.lookat - g_scene.view.eyepoint).getNormalized();
   Vec3d eye_right = (eye_dir^g_scene.view.up).getNormalized();
   Vec3d eye_up = eye_dir^eye_right*-1;

    // calculatehe dimensions of the viewport using the scene's camera
    float height = 2 * tan(M_PI/180 * .5 * g_scene.view.fovy);
    float width = height * g_scene.view.aspect;

    // compute delta steps in each direction
    Vec3d deltaX = eye_right * (width / xRes);
    Vec3d deltaY = eye_up * (height / yRes);

    // this should be bottom left
    Vec3d bottomLeft = g_scene.view.eyepoint + eye_dir - deltaX*xRes/2 - deltaY*yRes/2;

   // normal ray tracing: the color of the center of a pixel is computed
   #pragma omp parallel for schedule(dynamic) collapse(2)
   for (int sy=yRes ; sy > 0 ; --sy) {
      for (int sx=0 ; sx < xRes ; ++sx) {
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
               for(float dx = -1/4.; dx <= 1/4.; dx+=1/2.) {
                   for(float dy = -1/4.; dy <= 1/4.; dy+=1/2.) {
                       Vec3d superSamplePoint = point + deltaX*dx + deltaY*dy;
                       Vec3d superSampleDir = superSamplePoint - g_scene.view.eyepoint;

                       // create ray from view.eyepoint to view.lookat
                       Ray theRay(g_scene.view.eyepoint,superSampleDir.getNormalized(),0,g_objectList,g_lightList);
                       col += theRay.shade()*0.2;//color, recursive_ray_trace(eye, ray, 0));
                   }
               }
         }

         int index = 3*((sy-1) * xRes + sx);
         renderedScene[index + 0] = col[0];
         renderedScene[index + 1] = col[1];
         renderedScene[index + 2] = col[2];
      } // foreach x
   } // foreach y
}

void
Raytracer::startCuda(float *renderedScene, int xRes, int yRes)
{
    // create empty space for scene on gpu

    // updload current position of camera and the like, objects/lights/bgColor (should be done only once)

    // RAY TRACING:
    dim3 block(32, 16, 1);
    dim3 grid(xRes/ block.x, yRes / block.y, 1);
//    startCudaKernel<<<grid,block>>>block();
//startCudaKernel(float *renderedScene, int xRes, int yRes, Vec3d eyepoint, Vec3d up, Vec3d lookat, double aspect, bool d_antialiasing, Color backgroundCol)
}



void __global__
startCudaKernel(float *renderedScene, int xRes, int yRes, Vec3d eyepoint, Vec3d up, Vec3d lookat, double aspect, bool d_antialiasing, Color backgroundCol)
{
    // find out id of this thread
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx>=xRes*yRes)
        return;
    // we are handling pixel (x,y) now
    int sx = idx % xRes;
    int sy = idx / xRes;

    // setup viewport, its origin is bottom left
    // setup camera coordsys
    Vec3d eye_dir = (lookat - eyepoint).getNormalized();
    Vec3d eye_right = (eye_dir^up).getNormalized();
    Vec3d eye_up = eye_dir^eye_right*-1;

    // calculatehe dimensions of the viewport using the scene's camera
    float height = 2 * tan(M_PI/180 * .5 * fovy);
    float width = height * aspect;

    // compute delta steps in each direction
    Vec3d deltaX = eye_right * (width / xRes);
    Vec3d deltaY = eye_up * (height / yRes);

    // this should be bottom left
    Vec3d bottomLeft = eyepoint + eye_dir - deltaX*xRes/2 - deltaY*yRes/2;

    // the center of the pixel we are looking at right now
    Vec3d point = bottomLeft + deltaX*sx + deltaY*sy + deltaX/2 + deltaY/2;

    // the direction of our look
    Vec3d dir = point - eyepoint;

    // create ray from view.eyepoint to view.lookat
    Ray theRay(eyepoint,dir.getNormalized(),0,d_objectList,d_lightList);

    // compute the color
    Color col;
    theRay.shade(&theRay, eyepoint, point, d_objectList, objListSize, d_lightList, lightListSize, backgroundCol);

    // in case we are using antialiasing, calculate the color of this pixel by averaging
    if (d_antialiasing) {
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
        for(float dx = -1/4.; dx <= 1/4.; dx+=1/2.) {
            for(float dy = -1/4.; dy <= 1/4.; dy+=1/2.) {
                Vec3d superSamplePoint = point + deltaX*dx + deltaY*dy;
                Vec3d superSampleDir = superSamplePoint - eyepoint;

                // create ray from view.eyepoint to view.lookat
                Ray theRay(eyepoint,superSampleDir.getNormalized(),0,d_objectList,d_lightList);
                col += theRay.shade(&theRay,eyepoint,superSamplePoint, d_objectList, objListSize, d_lightList, lightListSize, backgroundCol)*0.2;
                //color, recursive_ray_trace(eye, ray, 0));
            }
        }
    }

    int index = 3*((sy-1) * xRes + sx);
    renderedScene[index + 0] = col[0];
    renderedScene[index + 1] = col[1];
    renderedScene[index + 2] = col[2];
}

