/** \file
 * \brief This module contains the OpenGL drawing routines.
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 1 - "Ui, OpenGL!"
 *
 * Created by Christian Vogelgsang <Vogelgsang@informatik.uni-erlangen.de>,
 * changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <ctime>
#include <cmath>

#include <QMouseEvent>
#include <QDebug>

#include <GL/glew.h>

#include "GLFrame.h"
#include "Shader.h"
#include "Model.h"


static const GLFrame::Material g_materials[] =
{
    // see http://devernay.free.fr/cours/opengl/materials.html
    {
        "Ruby",
        { 0.1745, 0.01175, 0.01175, 1.0},
        { 0.61424, 0.04136, 0.04136, 1.0},
        { 0.727811, 0.626959, 0.626959, 1.0},
        0.6
    },
    {
        "Copper",
        { 0.19125, 0.0735, 0.0225, 1.0},
        { 0.7038, 0.27048, 0.0828, 1.0},
        { 0.256777, 0.137622, 0.086014, 1.0},
        0.1
    },
    {
        "Gold",
        { 0.25, 0.20, 0.07, 1.0 },
        { 0.75, 0.60, 0.23, 1.0 },
        { 0.63, 0.56, 0.37, 1.0 },
        0.4
    },
    {
        "Brass",
        { 0.329412, 0.223529, 0.027451, 1.0 },
        { 0.780392	, 0.568627, 0.113725, 1.0 },
        { 0.992157, 0.941176, 0.807843	, 1.0 },
        0.21794872
    }
};

GLFrame::GLFrame(QWidget *parent)
: QGLWidget(parent)
, m_axesVisible(true)
, m_modelVisible(true)
, m_wireframeOverlayVisible(false)
, m_frameCounter(0)
, m_model(NULL)
, m_renderMode(WIRE_FRAME)
, m_glMode(DRAW_IMM)
, m_shaderType(SH_NONE)
, m_userParameter(1.0)
, m_currentMaterial(0)
{
    for(size_t i=0; i<sizeof(g_materials)/sizeof(g_materials[0]); ++i)
        m_materials.push_back(g_materials[i]);

    // set minium size of rendering area
    setMinimumSize(300,300);

    // setup OpenGL buffers
    QGLFormat format;
    // we only need a Depth buffer and a RGB frame buffer
    // double buffering is nice, too!
    format.setDoubleBuffer(true);
    format.setDepth(true);
    format.setRgba(true);
    format.setAlpha(true);
    format.setAccum(false);
    format.setStencil(false);
    format.setStereo(false);
    // try to enable format
    setFormat(format);

    memset(m_shader, '\0', sizeof(m_shader));

    // make the above state active
    resetCam();

    resetLight();
    m_time.start();
}

GLFrame::~GLFrame()
{
}

// ---------------------------- Basic OpenGL Widget Methods ------------------

// called by qt upon initialization of the GL window
void GLFrame::initializeGL()
{
    glewInit();

    glClearColor(0.25, 0.25, 0.25, 1.0); // let OpenGL clear to a dark grey
    glShadeModel(GL_FLAT );
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_DEPTH_TEST);

    loadShaders();
}


// called by qt whenever the window is resized
void GLFrame::resizeGL(int w, int h)
{
    m_aspect = (GLdouble)w / (GLdouble)h;
    m_width = w;
    m_height = h;

    glViewport(0, 0, (GLint)w, (GLint)h);
    updateGL();
}

// called by qt whenever repainting the window is necessary,
// mostly triggered by updateGL()
void GLFrame::paintGL()
{
    // start with a clean canvas
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // setup projection matrix
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    // adjust field of View
    gluPerspective(m_fieldOfView,m_aspect,0.01,100.0);

    // setup view matrix
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    // camera transformations
    glTranslatef(0,0,-m_distance);
    glMultMatrixf(m_camRot);

    // position light source
    // (push current position, move to light position,
    //  position light and then pop old position)
    glPushMatrix();
    glMultMatrixf(m_lightRot);
    glTranslatef(0., 0., m_lightDistance);

    GLfloat light_pos[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat white_light[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat lmodel_ambient[] = { 0.45, 0.45, 0.45, 1.0 };

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SINGLE_COLOR);
    glPopMatrix();

    // transform model to viewport center
    glPushMatrix();
    if(m_model)
    {
        const GLfloat scale = 3.0/m_model->size();
        glScalef(scale, scale, scale);
        GLfloat  center[3];
        m_model->getCenter(&center[0], &center[1], &center[2]);
        glTranslatef(-center[0], -center[1], -center[2]);
    }

    // draw coordinate axes
    if(m_axesVisible)
        drawCoordSys();

    // render the scene
    if(m_modelVisible)
    {
        const Material *mat = &m_materials[m_currentMaterial];

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat->ambient);
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat->diffuse);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat->specular);
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, mat->shininess*128.0);

        switch(m_renderMode)
        {
            case WIRE_FRAME:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                enableShader();
                drawModel(m_renderMode);
                disableShader();
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                break;
            case FLAT_SHADE:
            case SMOOTH_SHADE:
                enableLight();
                enableShader();
                drawModel(m_renderMode);
                disableShader();
                disableLight();
                break;
        }
    }

    if(m_wireframeOverlayVisible)
    {
        glColor3f(1.f, 0.f, 1.f);
        glPolygonOffset(0.0, -2.0);
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        drawModel(WIRE_FRAME);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    glPopMatrix();

    // draw transparent light source last
    glPushMatrix();
    glMultMatrixf(m_lightRot);
    glTranslatef(0., 0., m_lightDistance);
    drawLight();
    glPopMatrix();

    ++m_frameCounter;
}

// ------------------------ Interaction Handling -----------------------------

// qt notifies us that the mouse has been pressed
void GLFrame::mousePressEvent(QMouseEvent *m)
{
    m_lastMouseX = m->x();
    m_lastMouseY = m->y();
}

// dispatch mouse movement according to navMode and update scene
void GLFrame::mouseMoveEvent(QMouseEvent *m)
{
    int mouseButtons = m->buttons();
    if (!mouseButtons)
        return;

    int mouseX = m->x();
    int mouseY = m->y();

    const float dx = (mouseX - m_lastMouseX) / (float)m_width;
    // screen coordinates are the other way round -- swap again
    const float dy = (m_lastMouseY - mouseY) / (float)m_height;

    if(m->modifiers() && Qt::ShiftModifier)
        adjustLight(mouseButtons&Qt::LeftButton, mouseButtons&Qt::MidButton, mouseButtons&Qt::RightButton,
                  dx, dy);
    else
        adjustCam(mouseButtons&Qt::LeftButton, mouseButtons&Qt::MidButton, mouseButtons&Qt::RightButton,
                  dx, dy);

    m_lastMouseX = mouseX;
    m_lastMouseY = mouseY;

    // force openGL to redraw the changes
    updateGL();
}

void GLFrame::wheelEvent(QWheelEvent *ev)
{
    // a delta of 2880 is a full turn of the wheel
    if(ev->modifiers() & Qt::ShiftModifier)
        adjustLight(false, false, true, 0, -0.001*ev->delta());
    else
        adjustCam(false, false, true, 0, -0.001*ev->delta());

    updateGL();
}

static void matIdent(GLfloat *out)
{
    for(int i=0; i<16; i++)
    {
        out[i] = i%4==i/4 ? 1.0 : 0.0;
    }
}

static void matMult(GLfloat *out, const GLfloat *a, const GLfloat *b)
{
    // handle case of out==a or out==b
    GLfloat n[16];

    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            n[i*4+j] = 0.0;
            for(int k=0; k<4; k++) {
                n[i*4+j] += a[k*4+j]*b[i*4+k];
            }
        }
    }

    memcpy(out, n, sizeof(n));
}

static void matRot(GLfloat *out, const GLfloat *a, const GLfloat rad)
{
    matIdent(out);

    const float s = sin(rad);
    const float c = cos(rad);

    out[0] = a[0]*a[0]*(1-c)+c;
    out[1] = a[0]*a[1]*(1-c)-a[2]*s;
    out[2] = a[0]*a[2]*(1-c)+a[1]*s;

    out[4] = a[0]*a[1]*(1-c)+a[2]*s;
    out[5] = a[1]*a[1]*(1-c)+c;
    out[6] = a[1]*a[2]*(1-c)-a[0]*s;

    out[8] = a[0]*a[2]*(1-c)-a[1]*s;
    out[9] = a[1]*a[2]*(1-c)+a[0]*s;
    out[10] = a[2]*a[2]*(1-c)+c;
}

static bool matInv(GLfloat *inv, const GLfloat *mat)
{
    for(int i=0; i<16; i++)
        inv[i] = mat[i];

    int p[4];
    for (int k = 0; k < 4; k++)
    {
        p[k] = 0;
        float sup = 0.0;
        for (int i = k; i < 4; i++)
        {
            float s = 0.0;
            for (int j = k; j < 4; j++)
                s += fabs(inv[i*4+j]);
            float q = fabs(inv[i*4+k]) / s;
            if (sup < q)
            {
                sup = q;
                p[k] = i;
            }
        }
        if (fabs(sup) < 1e-6)
            return false;
        if (p[k] != k)
            for (int j = 0; j < 4; j++)
            {
                float h = inv[k*4+j];
                inv[k*4+j] = inv[p[k]*4+j];
                inv[p[k]*4+j] = h;
            }
        float pivot = inv[k*4+k];
        for (int j = 0; j < 4; j++)
            if (j != k)
            {
                inv[k*4+j] = -inv[k*4+j] / pivot;
                for (int i = 0; i < 4; i++)
                    if (i != k)
                        inv[i*4+j] += inv[i*4+k] * inv[k*4+j];
            }
        for (int i = 0; i < 4; i++)
            inv[i*4+k] = inv[i*4+k] / pivot;
        inv[k*4+k] = 1.0 / pivot;
    }

    for (int k = 4 - 1; k >= 0; k--)
        if (p[k] != k)
            for (int i = 0; i < 4; i++)
            {
                float h = inv[i*4+k];
                inv[i*4+k] = inv[i*4+p[k]];
                inv[i*4+p[k]] = h;
            }

    return true;
}



// interpret mouse movement in order to move the light
void GLFrame::adjustLight(bool leftButton, bool middleButton, bool rightButton,
                        float dx, float dy)
{
    if((leftButton && middleButton) || rightButton)
    {
        // prevent negative distance
        if(dy < -0.5)
            dy = 0.5;

        m_lightDistance *= 1.0+dy;
    }
    else if(leftButton)
    {
        float d = sqrt(dx*dx+dy*dy); // distance of incremental mouse movement

        if(d > 0.0001)
        {
            // incremental transformation
            GLfloat a[4] = {dy/d, -dx/d, 0.0, 0.0}; // rotation axis
            GLfloat inc[16], invCam[16];
            matInv(invCam, m_camRot);
            matRot(inc, a, d*10);
            matMult(inc, inc, m_camRot);
            matMult(inc, invCam, inc);

            // multiply viewing transformation with incremental transformation
            // to become the new viewing transformation
            matMult(m_lightRot, inc, m_lightRot);
        }
    }
    else if(middleButton)
    {
        GLfloat inc[16], invCam[16];
        matInv(invCam, m_camRot);
        matIdent(inc);
        inc[12] += dx*10.;
        inc[13] += dy*10.;
        matMult(inc, inc, m_camRot);
        matMult(inc, invCam, inc);
        matMult(m_lightRot, inc, m_lightRot);
    }
}

// interpret mouse movement in order to move the camera or adjust
// its field of view
void GLFrame::adjustCam(bool leftButton, bool middleButton, bool rightButton,
                        float dx, float dy)
{
    if(leftButton && middleButton)
    {
        // prevent negative distance
        if(dy < -0.5)
            dy = 0.5;

        m_distance *= 1.0+dy;
    }
    else if(leftButton)
    {
        float d = sqrt(dx*dx+dy*dy); // distance of incremental mouse movement

        if(d > 0.0001)
        {
            float a[4] = {dy/d, -dx/d, 0.0, 0.0}; // rotation axis
            GLfloat inc[16];
            matRot(inc, a, d*10);

            // multiply viewing transformation with incremental transformation
            // to become the new viewing transformation
            matMult(m_camRot, inc, m_camRot);
        }
    }
    else if(middleButton)
    {
        m_camRot[12] += dx*10.;
        m_camRot[13] += dy*10.;
    }
    else if(rightButton)
    {
        // prevent negative fieldOfView
        if(dy < -0.5)
            dy = 0.5;

        m_fieldOfView *= 1.0+dy;
        if(m_fieldOfView > 160.0)
            m_fieldOfView = 160.0;
    }
}

// notify when the drawMode should be changed
void GLFrame::setRenderMode(int mode)
{
    m_renderMode = (RenderMode)mode;
    updateGL();
}

void GLFrame::setWireframeOverlayVisibility(bool on)
{
    m_wireframeOverlayVisible = on;
    updateGL();
}

void GLFrame::setModelVisibility(bool on)
{
    m_modelVisible = on;
    updateGL();
}

void GLFrame::setAxesVisibility(bool on)
{
    m_axesVisible = on;
    updateGL();
}

// reset camera to default values
void GLFrame::resetLight()
{
    m_lightDistance = 3.0;

    matIdent(m_lightRot);

    updateGL();
}

// reset camera to default values
void GLFrame::resetCam()
{
    m_distance = 10.0;

    matIdent(m_camRot);

    m_fieldOfView = 15.0;
    updateGL();
}

// ------------------------- Drawing Methods ---------------------------------

// draw the three coord axes
void GLFrame::drawCoordSys()
{
    // draw coordinate axes
    glBegin(GL_LINES);

    // x: red
    glColor3f(1.0,0.0,0.0);
    glVertex3f(0.,0.,0.);
    glVertex3f(1.,0.,0.);

    // y: green
    glColor3f(0.0,1.0,0.0);
    glVertex3f(0.,0.,0.);
    glVertex3f(0.,1.,0.);

    // z: blue
    glColor3f(0.0,0.0,1.0);
    glVertex3f(0.,0.,0.);
    glVertex3f(0.,0.,1.);

    glEnd();
}

void GLFrame::drawLight()
{
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glColor4f(1.0, 1.0, 0.0, 0.5);
    GLUquadric *q = gluNewQuadric();
    gluSphere(q, 0.1, 15, 15);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
}

// draw model
void GLFrame::drawModel(RenderMode mode)
{
    if(!m_model)
        return;

    switch(m_glMode)
    {
        case DRAW_IMM:
            m_model->drawImmediate(mode==SMOOTH_SHADE);
            break;
        case DRAW_ARRAY:
            m_model->drawArray(mode==SMOOTH_SHADE);
            break;
        case DRAW_VBO:
            m_model->drawVBO(mode==SMOOTH_SHADE);
            break;
        case NUM_DRAW:
            break;
    }
}

// enable lighting
void GLFrame::enableLight()
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_RESCALE_NORMAL);
    glShadeModel(GL_SMOOTH);
}

// switch the light off
void GLFrame::disableLight()
{
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_RESCALE_NORMAL);
    glShadeModel(GL_FLAT);
}


void GLFrame::loadModel(const QString &model)
{
    delete m_model;

    m_model = new Model(this);
    connect(m_model, SIGNAL(info(QString)), this, SIGNAL(info(QString)));
    m_model->read(model);
    updateGL();
}

int GLFrame::frameCounter() const
{
    return m_frameCounter;
}
    
bool GLFrame::loadShaders()
{
    bool ok = false;
    for(int i=0; i<NUM_SHS; ++i)
    {
        m_shader[i] = new Shader();
        switch(i)
        {
            case SH_SIMPLE:
                ok |= m_shader[i]->loadVertexSource("simple.vsh");
                ok |= m_shader[i]->loadFragmentSource("simple.fsh");
                break;
            case SH_PHONG:
                ok |= m_shader[i]->loadVertexSource("phong.vsh");
                ok |= m_shader[i]->loadFragmentSource("phong.fsh");
                break;
            case SH_FREESTYLE:
                ok |= m_shader[i]->loadVertexSource("freestyle.vsh");
                ok |= m_shader[i]->loadFragmentSource("freestyle.fsh");
                break;
        }
        ok |= m_shader[i]->link();
    }
    return ok;
}

bool GLFrame::unloadShaders()
{
    bool ok = false;
    for(int i=0; i<NUM_SHS; ++i)
    {
        delete m_shader[i];
    }
    return ok;
}

bool GLFrame::enableShader()
{
    bool ok = true;
    if(m_shaderType==SH_NONE)
    {
        glUseProgram(0);
    }
    else
    {

        const double sec = m_time.elapsed() * 0.001;
        m_shader[m_shaderType]->enable();
        m_shader[m_shaderType]->setUniform1f("Time", sec);
        m_shader[m_shaderType]->setUniform1f("UserParameter", m_userParameter);
    }

    return ok;
}

bool GLFrame::disableShader()
{
    bool ok = true;
    if(m_shaderType==SH_NONE)
    {
        glUseProgram(0);
    }
    else
        m_shader[m_shaderType]->disable();
    return ok;
}

void GLFrame::setGlMode(int technique)
{
    if(technique >= 0 && technique < NUM_DRAW)
        m_glMode = (GLMode)technique;
    updateGL();
}

void GLFrame::setShader(int shader)
{
    if(shader >= SH_NONE && shader < NUM_SHS)
        m_shaderType = (ShaderType)shader;
    updateGL();
}

void GLFrame::setUserParameter(double param)
{
    m_userParameter = param;
    updateGL();
}

int GLFrame::numMaterials() const
{
    return m_materials.size();
}

int GLFrame::currentMaterial() const
{
    return m_currentMaterial;
}

const char *GLFrame::materialName(int index) const
{
    if(index < 0 || index >= numMaterials())
        return NULL;

    return m_materials[index].name;
}

void GLFrame::setMaterial(int index)
{
    if(index >= 0 && index < numMaterials())
        m_currentMaterial = index;

    updateGL();
}
