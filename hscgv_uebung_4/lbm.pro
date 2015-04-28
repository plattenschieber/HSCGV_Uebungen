QT       += core gui opengl

TARGET = lbm
TEMPLATE = app

# use this for an optimised build
#CONFIG += debug
CONFIG += release

# Adjust this to point to your OpenSceneGraph folder
OSGHOME=openscenegraph

OSG_LIBS = -losgText -losgGA -losgFX \
    -losgDB -losgUtil -losg \
    -lOpenThreads -losgViewer

INCLUDEPATH *= $${OSGHOME}/include
LIBS *= -L$${OSGHOME}/lib $${OSG_LIBS}

CUDA_SOURCES += \
    cudaTest.cu \
    lbm.cu

SOURCES += main.cpp \
    applicationwindow.cpp \
    lbm.cpp \
    coordinatebox.cpp \
    slicevisualisation.cpp \
    linevisualisation.cpp \
    slice.cpp \
    clock.cpp \
    qosgviewer.cpp \
    overlayviewer.cpp \
    helpdialog.cpp

HEADERS += \
    applicationwindow.h \
    lbm.h \
    coordinatebox.h \
    slicevisualisation.h \
    linevisualisation.h \
    slice.h \
    clock.h \
    qosgviewer.h \
    overlayviewer.h \
    helpdialog.h \
    cudaTest.h

release {
# in release mode, use OpenMP parallelized CPU code
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
}

include(cuda.pri)
