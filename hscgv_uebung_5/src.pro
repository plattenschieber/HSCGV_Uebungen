TARGET=         raytrace
QT *=  opengl

HEADERS=        geoobject.h geoquadric.h lightobject.h ray.h types.h vector.h param.h \
    geopolygon.h \
    ApplicationWindow.h \
    ui_ApplicationWindow.h
SOURCES=        geoobject.cpp geoquadric.cpp lightobject.cpp main.cpp ray.cpp \
    geopolygon.cpp \
    ApplicationWindow.cpp
win32:SOURCES*= xgetopt.cpp
YACCSOURCES=    input.y

QMAKE_YACC              = bison
QMAKE_YACCFLAGS         = -y -d

# comment these for a release version
DEFINES         *= TRACE VERBOSE
CONFIG          += debug
CONFIG          -= release

QMAKE_CXXFLAGS += -W -Wall

FORMS += \
    ApplicationWindow.ui
