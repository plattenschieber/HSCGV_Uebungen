TARGET=         raytrace
QT *=  opengl

HEADERS=        geoobject.h geoquadric.h lightobject.h ray.h types.h vector.h param.h \
    geopolygon.h \
    ApplicationWindow.h \
    GLFrame.h \
    raytracer.h
SOURCES=        geoobject.cpp geoquadric.cpp lightobject.cpp main.cpp ray.cpp \
    geopolygon.cpp \
    ApplicationWindow.cpp \
    GLFrame.cpp \
    raytracer.cpp
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

unix:!macx:LIBS *= -lGLEW -lglut -lGLU

macx {
LIBS *= -lGLEW -L/usr/local/opt/glew/lib
INCLUDEPATH *= /usr/local/opt/glew/include
}

win32 {
# adapt to your needs
GLEWDIR = c:/glew
INCLUDEPATH *= $$GLEWDIR/include
DEFINES *= GLEW_STATIC
LIBS *= -L$$GLEWDIR/lib -lglew32s
}
