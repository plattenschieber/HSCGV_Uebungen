#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#ifdef _WIN32
#include "xgetopt.h"
#else
#include <getopt.h>
#endif
#include "geoobject.h"
#include "lightobject.h"
#include "ray.h"
#include "param.h"
#include "types.h"

#ifndef M_PI
#define M_PI 3.1415927
#endif

class Raytracer
{
public:
    Raytracer();
    Raytracer(bool antialiasing);
    Raytracer(const char* filename, bool antialiasing);
    ~Raytracer();
    void render(float* renderedScene, int xRes, int yRes);
    void renderCuda(float *renderedScene, int xRes, int yRes);
    void initCuda();
    bool m_isFileLoaded;

    //! CUDA Properties
    GeoObject* d_objList;
    GeoObjectProperties* d_objPropList;
    LightObject* d_lightList;
    LightObjectProperties* d_lightPropList;
    float *d_renderedScene;

private:
    const char *m_filename;
    bool m_antialiasing;
};

// Description:
// the connection to the parser
extern int openSceneFile(const char *s);
extern int closeSceneFile(void);
extern int inputparse(void);
extern int cleanUp(void);
// scene storage
//! the list of geometric objects in the scene
extern std::vector<GeoObject*>           g_objectList;
//! the list of light sources in the scene
extern std::vector<LightObject*>         g_lightList;
//! the list of surface properties of the objects
extern std::vector<GeoObjectProperties*> g_propList;

#endif // RAYTRACER_H
