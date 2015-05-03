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

#include <QGLWidget>
#include <QTime>

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
    GLFrame(QWidget *parent=0);
    //! destroy an OGL Frame
    virtual ~GLFrame();

    //! the drawing modes for the model
    enum RenderMode {CPU, GPU};

    //! accessor for frame counter
    int frameCounter() const;

    // (Some Slots and Signals)
signals:
    //! show some information to the user
    void info(QString msg);

public slots:
    //! change render mode (wireframe, ...)
    void setRenderMode(int mode);
    //! set visibility of coordinate axes
    void setAxesVisibility(bool on);
    //! reset camera to default values
    void resetCam();
    //! reset light to default values
    void resetLight();
    //! read model file
    void loadModel(const QString &filename);
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

    //! draw model
    void drawModel(RenderMode mode);

    //! draw a representation for the light source
    void drawLight();

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
    
    //! axes visibility
    bool m_axesVisible;
    //! model visibility
    bool m_modelVisible;

    //! counter for displaying FPS
    unsigned int m_frameCounter;

    //! currently loaded model
    GeoObject *m_model;

    //! how our model is to be drawn
    RenderMode m_renderMode;

    //! timer for Time shader parameter
    QTime m_time;

};

#endif
