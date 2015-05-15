/** \file
 * \brief Declaration of the OpenGL View Frame
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 1 - "Ui, OpenGL!"
 *
 * Created by Christian Vogelgsang <Vogelgsang@informatik.uni-erlangen.de>,
 * changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef GLFRAME_H
#define GLFRAME_H

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#elif
#include <GL/glew.h>
#include <GL/glu.h>
#endif
#include <QGLWidget>
#include <QTime>
#include <qdebug.h>
#include "raytracer.h"

Q_DECLARE_METATYPE(unsigned char *)

#ifdef NDEBUG
#define CHECKGL
#else
inline bool checkGL(const char *label, int line)
{
    GLenum err = glGetError();
    if(err == GL_NO_ERROR)
        return true;

    qDebug() << "GL error -" << label << line << (char*)gluErrorString(err);
    return false;
}
#ifdef _WIN32
#define CHECKGL do { checkGL(__FUNCTION__, __LINE__); } while(0)
#else
#define CHECKGL do { checkGL(__PRETTY_FUNCTION__, __LINE__); } while(0)
#endif
#endif

#include "ApplicationWindow.h"
class ApplicationWindow;

class QMouseEvent;

class GeoObject;

// ----- GLFrame -----

/** Our own OpenGL drawing area, derived from QGLWidget,
 * it loads a PLY model from file and displays it corresponding to the current options
 */
//! OpenGL drawing area, derived from QGLWidget, loads and displays a PLY model
class GLFrame : public QGLWidget
{
    // include meta object information
    Q_OBJECT

public:
    //! create a new GLFrame
    GLFrame(ApplicationWindow *parent=0);
    //! destroy an OGL Frame
    virtual ~GLFrame();

    //! the drawing modes for the model
    enum RenderMode {CPU, GPU};

    //! accessor for frame counter
    int frameCounter() const;

    //! antialiasing
    bool m_antialiasing;

    // (Some Slots and Signals)
signals:
    //! show some information to the user
    void info(QString msg);

public slots:
    //! change render mode (wireframe, ...)
    void setRenderMode(int mode);
    //! set visibility of coordinate axes
    void setAxesVisibility(bool on);
    //! set antialising
    void setAntialiasing(bool on);
    //! reset camera to default values
    void resetCam();
    //! reset light to default values
    void resetLight();
    //! read scene file
    void loadScene(const QString &filename);
    //! load the viewport, camera, position etc. into this frame
    void loadViewFromData();
protected:

    // ---- OGLWidget Basic Methods ----

    //! setup the openGL state machine for this render context
    void initializeGL();

    //! the viewport has been resized -> notify the widget.
    void resizeGL(int w, int h);

    //! draw the entire scene
    void paintGL();

    // ---- QWidget Event Handling ----

    //! the mouse has been pressed
    void mousePressEvent(QMouseEvent *);

    //! the mouse has been moved
    void mouseMoveEvent(QMouseEvent *);

    //! mouse wheel has been turned
    void wheelEvent(QWheelEvent *);

    //! draw the coordinate system with OpenGL
    void drawCoordSys();

    //! draw a representation for the light source
    void drawLight();

    //! draws a quad where we will paint the raytracing scene in a texture
    void drawFullScreenQuad();

    //! draw scene
    void renderScene(RenderMode mode);

    //! load current raytracing scene into texture
    void loadTexture();

    // ---- More Drawing and Interaction Handling ----

    //! adjust the camera according to the mouse action
    void adjustCam(bool leftButton, bool middleButton, bool rightButton,
                   float dx, float dy);

    //! adjust the light according to the mouse action
    void adjustLight(bool leftButton, bool middleButton, bool rightButton,
                     float dx, float dy);

    // ----- internal state -----

    // the parameters of the current view port
    //! viewport width
    int m_width;
    //! viewport height
    int m_height;
    //! viewport aspect ratio (determined by window size)
    GLfloat m_aspect;
    //! field of view
    GLfloat m_fieldOfView;

    //! State variables for Interaction
    //! mouse x coordinate is saved for calculating the movement
    int m_lastMouseX;
    //! mouse y coordinate is saved for calculating the movement
    int m_lastMouseY;
    //! camera position
    GLfloat m_camRot[16];
    //! camera distance
    GLfloat m_distance;

    //! light position
    GLfloat m_lightRot[16];

    //! position of the light source
    GLfloat m_lightDistance;

    // what to draw
    
    //! the currently used texture cache
    GLuint m_texHandle;
    //! axes visibility
    bool m_axesVisible;
    //! model visibility
    bool m_modelVisible;

    //! counter for displaying FPS
    unsigned int m_frameCounter;

    //! currently loaded scene
    Raytracer *m_raytracer;
    bool m_raytracingNeeded;
    float* m_data;

    //! how our model is to be drawn
    RenderMode m_renderMode;

    //! timer for Time shader parameter
    QTime m_time;
};

#endif
