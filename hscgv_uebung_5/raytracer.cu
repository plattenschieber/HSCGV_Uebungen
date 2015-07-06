/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "raytracer.h"
#ifndef __APPLE__
#include "omp.h"
#endif

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
Raytracer::render(float *renderedScene, int xRes, int yRes) {
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void __global__
initPropertiesKernel(GeoObject* d_objList, GeoObjectProperties* d_objPropList, int d_objListSize, LightObject* d_lightList, LightObjectProperties* d_lightPropList, int d_lightListSize) {
    // setup the objects properties one by one, since we needed to copy them by hand into different lists (objPropList and lightPropList)
    for (int i=0; i<d_objListSize; i++)
        d_objList[i].setProperties(&d_objPropList[i]);
    for (int i=0; i<d_lightListSize; i++)
        d_lightList[i].setProperties(&d_lightPropList[i]);
}

void __global__
renderKernel(float *d_renderedScene, int xRes, int yRes, Vec3d eyepoint, Vec3d up, Vec3d lookat, double aspect, double fovy, Color backgroundCol, bool antialiasing,
                GeoObject* d_objList, int objListSize, LightObject* d_lightList, int lightListSize)
{
    // find out id of this thread
    unsigned sx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= xRes || sy >= yRes)
        return;

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
    Ray theRay(eyepoint,dir.getNormalized(),0);

    // compute the color
    Color col;
    theRay.shade(&theRay, eyepoint, point, d_objList, objListSize, d_lightList, lightListSize, backgroundCol);

    // in case we are using antialiasing, calculate the color of this pixel by averaging
    if (antialiasing) {
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
                Ray theRay(eyepoint,superSampleDir.getNormalized(),0);
                col += theRay.shade(&theRay,eyepoint,superSamplePoint, d_objList, objListSize, d_lightList, lightListSize, backgroundCol)*0.2;
                //color, recursive_ray_trace(eye, ray, 0));
            }
        }
    }

    int index = 3*((sy-1) * xRes + sx);
    d_renderedScene[index + 0] = col[0];
    d_renderedScene[index + 1] = col[1];
    d_renderedScene[index + 2] = col[2];
}



//! we need some kind of initialization of our device
void
Raytracer::initCuda() {
    // get some space for the objects and their properties (
    gpuErrchk (cudaMalloc((void **) &d_objList, sizeof(GeoObject) * g_objectList.size()));
    gpuErrchk (cudaMalloc((void **) &d_objPropList, sizeof(GeoObjectProperties) * g_objectList.size()));
    gpuErrchk (cudaMalloc((void **) &d_lightList, sizeof(LightObject) * g_lightList.size()));
    gpuErrchk (cudaMalloc((void **) &d_lightPropList, sizeof(LightObjectProperties) * g_lightList.size()));

    gpuErrchk (cudaMemcpy(d_objList, g_objectList.data(), sizeof(GeoObject) * g_objectList.size(), cudaMemcpyHostToDevice));
    gpuErrchk (cudaMemcpy(d_objPropList, g_objectList.data(), sizeof(GeoObjectProperties) * g_objectList.size(), cudaMemcpyHostToDevice));
    gpuErrchk (cudaMemcpy(d_lightList, g_objectList.data(), sizeof(LightObject) * g_lightList.size(), cudaMemcpyHostToDevice));
    gpuErrchk (cudaMemcpy(d_lightPropList, g_objectList.data(), sizeof(LightObjectProperties) * g_lightList.size(), cudaMemcpyHostToDevice));

    initPropertiesKernel<<<1,1>>>(d_objList, d_objPropList, g_objectList.size(), d_lightList, d_lightPropList, g_lightList.size());
 }

//-------------------------------------------------------------------------------------------------
// Round (a) up to the nearest multiple of (b), then divide by (b)
//

int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

//! start the rendering routine on the device
void
Raytracer::renderCuda(float *cudaData, int xRes, int yRes)
{
    // RAY TRACING:
    initCuda();
    dim3 block(16, 16);
    dim3 grid(div_up(xRes, block.x), div_up(yRes, block.y));

    renderKernel<<<grid,block>>>(cudaData, xRes, yRes,
                                    g_scene.view.eyepoint, g_scene.view.up, g_scene.view.lookat, g_scene.view.aspect, g_scene.view.fovy, g_scene.picture.background,
                                    m_antialiasing, d_objList, g_objectList.size(), d_lightList, g_lightList.size());
}
