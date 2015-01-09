TARGET = gl

FORMS = ApplicationWindow.ui UserParameterDialog.ui
HEADERS = ApplicationWindow.h UserParameterDialog.h GLFrame.h ply.h Shader.h \
    Model.h
SOURCES = main.cpp ApplicationWindow.cpp UserParameterDialog.cpp GLFrame.cpp ply.c Shader.cpp \
    Model.cpp
OTHER_FILES += \
    simple.vsh \
    simple.fsh \
    phong.vsh \
    phong.fsh \
    freestyle.vsh \
    freestyle.fsh

unix:LIBS *= -lGLEW

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

QT *=  opengl

# this does not play nice with Qt Creator
#CONFIG *= debug
#CONFIG -= release
