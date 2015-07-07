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

// Description:
// Return the normal vector at point v of this surface
// TODO this is an object method, update such that the called on object is included
Vec3d __device__
cudaGetNormal(const Vec3d &v, const QUADRIC &q)
{
   Vec3d tmpvec( (v | Vec3d(2*q.m_a,q.m_b,q.m_c)) + q.m_d,
         (v | Vec3d(q.m_b,2*q.m_e,q.m_f)) + q.m_g,
         (v | Vec3d(q.m_c,q.m_f,2*q.m_h)) + q.m_j );
   return tmpvec.getNormalized();
}
// Description:
// Compute intersection point on this surface
// return distance to nearest intersection found or -1
// TODO this is an object method, update such that the called on object is included
double __device__
cudaIntersect(const RAY &ray, QUADRIC &q)
{
   double t = -1.0, acoef, bcoef, ccoef, root, disc;

   acoef = Vec3d( (ray.m_direction | Vec3d(q.m_a,q.m_b,q.m_c)),
         q.m_e * ray.m_direction[1] + q.m_f * ray.m_direction[2],
         q.m_h * ray.m_direction[2])
      | ray.m_direction;

   bcoef =   (Vec3d(q.m_d,q.m_g,q.m_j) | ray.m_direction)
      + (ray.m_origin | Vec3d((ray.m_direction | Vec3d(2*q.m_a,q.m_b,q.m_c)),
               (ray.m_direction | Vec3d(q.m_b,2*q.m_e,q.m_f)),
               (ray.m_direction | Vec3d(q.m_c,q.m_f,2*q.m_h))));

   ccoef = (ray.m_origin | Vec3d((Vec3d(q.m_a,q.m_b,q.m_c) | ray.m_origin) + q.m_d,
            q.m_e * ray.m_origin[1] + q.m_f * ray.m_origin[2] + q.m_g,
            q.m_h * ray.m_origin[2] + q.m_j))
      + q.m_k;

   if (acoef != 0.0) {
      disc = bcoef * bcoef - 4.0 * acoef * ccoef;
      if (disc > -Vec3d::getEpsilon()) {
         root = sqrt( disc );
         t = ( -bcoef - root ) / ( acoef + acoef );
         if (t < 0.0)
            t = ( -bcoef + root ) / ( acoef + acoef );
      }
   }
   return (((Vec3d::getEpsilon() * 10.0) < t) ? t : -1.0);
}
// Description:
// Determine color contribution of a lightsource
Color __device__
cudaShadedColor(LIGHT *light, const RAY &reflectedRay, const Vec3d &normal, QUADRIC *obj)
{
   double ldot = light->m_direction | normal;
   Color reflectedColor = Color(0.0);

   // lambertian reflection model
   if (ldot > 0.0)
      reflectedColor += obj->m_reflectance * (light->m_color * ldot);

   // updated with ambient lightning as in:
   // [GENERALISED AMBIENT REFLECTION MODELS FOR LAMBERTIAN AND PHONG SURFACES, Xiaozheng Zhang and Yongsheng Gao]
//   reflectedColor += obj->ambient() * g_sceneCuda.ambience;

   // specular part
   double spec = reflectedRay.m_direction | light->m_direction;
   if (spec > 0.0) {
      spec = obj->m_specular * pow(spec, obj->m_specularExp);
      reflectedColor += light->m_color * spec;
   }

   return Color(reflectedColor);
}


Color __device__
cudaShade(RAY *thisRay, Vec3d d_origin, Vec3d d_direction, QUADRIC *d_objList, int objListSize, LIGHT *d_lightList, int lightListSize, Color background)
{
    Color currentColor(0.0);
    for (int i=0; i<5; i++) {
        QUADRIC *closest = NULL;
        double tMin = DBL_MAX;

        // find closest object that intersects
        for (int j=0; j<objListSize; j++)
        {
            double t = cudaIntersect(*thisRay, d_objList[j]);
            if (0.0 < t && t < tMin) {
                tMin = t;
                closest = &d_objList[j];
}
        }

        // no object hit -> ray goes to infinity
        if (closest == NULL) {
            if (i == 0) {
                return background; // background color
            }
            else {
                return Color(0.0);         // black
            }
        }
        else {
            // reflection
            Vec3d intersectionPosition(d_origin + (d_direction * tMin));
            Vec3d normal(cudaGetNormal(intersectionPosition, *closest));
            RAY reflectedRay;
            reflectedRay.m_origin = intersectionPosition;
            reflectedRay.m_direction = d_direction.getReflectedAt(normal).getNormalized();
            reflectedRay.m_depth = i+1;

            // calculate lighting
            for (int j=0; j<lightListSize; j++) {

                // where is the lightsource ?
                RAY rayoflight;
                rayoflight.m_origin = intersectionPosition;
                rayoflight.m_direction = d_lightList[j].m_direction;
                rayoflight.m_depth = 0;
                bool something_intersected = false;

                // where are the objects ?
                for (int k=0; k<objListSize; k++) {

                    double t = cudaIntersect(rayoflight, d_objList[k]);
                    if (t > 0.0) {
                        something_intersected = true;
                        break;
                    }

                } // for all obj

                // is it visible ?
                if (! something_intersected)
                    currentColor += cudaShadedColor(&d_lightList[j], reflectedRay, normal, closest);

            } // for all lights

            // could be right...
            currentColor *= closest->m_mirror;
        }
   }
   return Color(currentColor);
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
renderKernel(float *d_renderedScene, int xRes, int yRes, Vec3d eyepoint, Vec3d up, Vec3d lookat, double aspect, double fovy, Color backgroundCol, bool antialiasing,
                QUADRIC* d_objList, int objListSize, LIGHT* d_lightList, int lightListSize)
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
    RAY theRay;
    theRay.m_origin = eyepoint;
    theRay.m_direction = dir.getNormalized();
    theRay.m_depth = 0;

    // compute the color
    Color col;
    cudaShade(&theRay, eyepoint, point, d_objList, objListSize, d_lightList, lightListSize, backgroundCol);

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
                RAY theRayA;
                theRayA.m_origin = eyepoint;
                theRayA.m_direction = superSampleDir.getNormalized();
                theRayA.m_depth = 0;
                col += cudaShade(&theRay,eyepoint,superSamplePoint, d_objList, objListSize, d_lightList, lightListSize, backgroundCol)*0.2;
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
    initPropertiesKernel<<<1,1>>>(d_objList, d_objPropList, g_objectList.size(), d_lightList, d_lightPropList, g_lightList.size());
    gpuErrchk (cudaMalloc((void **) &d_objList, sizeof(QUADRIC) * g_objectList.size()));
    gpuErrchk (cudaMalloc((void **) &d_lightList, sizeof(LIGHT) * g_lightList.size()));

    gpuErrchk (cudaMemcpy(d_objList, quads.data(), sizeof(QUADRIC) * g_objectList.size(), cudaMemcpyHostToDevice));
    gpuErrchk (cudaMemcpy(d_lightList, lights.data(), sizeof(LIGHT) * g_lightList.size(), cudaMemcpyHostToDevice));
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
