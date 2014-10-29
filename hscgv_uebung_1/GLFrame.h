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

class Shader;
class Model;

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
    enum RenderMode { WIRE_FRAME, FLAT_SHADE, SMOOTH_SHADE};
    //! the GL render techniques
    enum GLMode { DRAW_IMM, DRAW_ARRAY, DRAW_VBO, NUM_DRAW };
    //! the GLSL shaders
    enum ShaderType { SH_NONE=-1, SH_SIMPLE, SH_PHONG, SH_FREESTYLE, NUM_SHS };

    //! accessor for frame counter
    int frameCounter() const;

    //! material properties
    struct Material
    {
        //! name to display in menu
        const char *name;
        //! ambient component
        float ambient[4];
        //! diffuse component
        float diffuse[4];
        //! specular component
        float specular[4];
        //! specular exponent
        float shininess;
    };
    //! number of available materials
    int numMaterials() const;
    //! currently used material
    int currentMaterial() const;
    //! name for material with number index
    const char *materialName(int index) const;

    // (Some Slots and Signals)
signals:
    //! show some information to the user
    void info(QString msg);

public slots:
    //! change material
    void setMaterial(int index);
    //! change render mode (wireframe, ...)
    void setRenderMode(int mode);
    //! change GL render technique
    void setGlMode(int technique);
    //! load a different GLSL shader
    void setShader(int shader);
    //! notify when visibility of axes/model should be changed
    void setModelVisibility(bool on);
    //! set visibility of coordinate axes
    void setAxesVisibility(bool on);
    //! switch visibility of wireframe overlay
    void setWireframeOverlayVisibility(bool on);
    //! reset camera to default values
    void resetCam();
    //! reset light to default values
    void resetLight();
    //! read model file
    void loadModel(const QString &filename);
    //! update user defined shader paraemter
    void setUserParameter(double param);
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

    //! switch the light on
    void enableLight();

    //! switch the light off
    void disableLight();

    // ---- More Drawing and Interaction Handling ----

    //! adjust the camera according to the mouse action
    void adjustCam(bool leftButton, bool middleButton, bool rightButton,
                   float dx, float dy);

    //! adjust the light according to the mouse action
    void adjustLight(bool leftButton, bool middleButton, bool rightButton,
                     float dx, float dy);

    //! load all shaders
    bool loadShaders();
    //! delete all shaders
    bool unloadShaders();
    //! enable current shader
    bool enableShader();
    //! disable all shaders
    bool disableShader();

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
    //! wireframe overlay visibility
    bool m_wireframeOverlayVisible;

    //! counter for displaying FPS
    unsigned int m_frameCounter;

    //! currently loaded model
    Model *m_model;

    //! how our model is to be drawn
    RenderMode m_renderMode;

    //! drawing technique
    GLMode m_glMode;

    //! index of current shader
    ShaderType m_shaderType;
    //! available shaders
    Shader *m_shader[NUM_SHS];

    //! value of user defined paramter
    GLfloat m_userParameter;
    //! timer for Time shader parameter
    QTime m_time;

    //! list if materials
    std::vector<Material> m_materials;
    //! index of current material
    int m_currentMaterial;
};

#endif
