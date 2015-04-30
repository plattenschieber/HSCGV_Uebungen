QT       += core gui opengl

TARGET=         raytrace
TEMPLATE =      app

CUDA_SOURCES += cudaTest.cu

HEADERS=        geoobject.h geoquadric.h lightobject.h ray.h types.h vector.h param.h \
    geopolygon.h \
    cudaTest.h
SOURCES=        geoobject.cpp geoquadric.cpp lightobject.cpp main.cpp ray.cpp \
    geopolygon.cpp
win32:SOURCES*= xgetopt.cpp
YACCSOURCES=    input.y

QMAKE_YACC              = bison
QMAKE_YACCFLAGS         = -y -d

# we don't need Qt
CONFIG          -= qt
# this is an application to be run from the command line
CONFIG          *= console
CONFIG          -= app_bundle

# comment these for a release version
DEFINES         *= TRACE VERBOSE
CONFIG          += debug
CONFIG          -= release

QMAKE_CXXFLAGS += -W -Wall -Wextra -pedantic

include(cuda.pri)
