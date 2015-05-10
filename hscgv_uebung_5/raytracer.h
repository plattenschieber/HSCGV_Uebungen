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
    void start();
    bool m_isFileLoaded;
private:
    const char *m_filename;
    bool m_antialiasing;
};

#endif // RAYTRACER_H
