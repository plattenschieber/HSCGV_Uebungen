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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "GLFrame.h"


GLFrame::GLFrame(ApplicationWindow *parent)
: QGLWidget(parent)
, m_antialiasing(false)
, m_axesVisible(true)
, m_modelVisible(true)
, m_frameCounter(0)
, m_raytracer(NULL)
, m_raytracingNeeded(false)
, m_renderMode(CPU)
{
    // set minium size of rendering area
    setMinimumSize(150,150);

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

    // make the above state active
    resetCam();

    resetLight();
    m_time.start();
}

GLFrame::~GLFrame()
{
    glDeleteTextures(1, &m_texHandle);
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
    // allocate a texture name
    glGenTextures( 1, &m_texHandle );CHECKGL;
}

void GLFrame::loadTexture()
{
    // select our current texture
    glBindTexture( GL_TEXTURE_2D, m_texHandle );
    // important, since we store data in a contignous array (default is 4 bytes alignment)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // when texture area is small, bilinear filter the closest MIP map
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // when texture area is large, bilinear filter the first MIP map
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // the texture ends at the edges (clamp)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    // move data into a texture - the last parameter is either host data or
    // NULL since we only want to allocate memory, not initialize it in case of PBO
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_FLOAT,
//                 (m_renderMode==CPU)?((GLvoid*)m_data):NULL); CHECKGL;
    // move data into a texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_FLOAT, (GLvoid*)m_data); CHECKGL;

    // free data since its on the texture
    free(m_data);

//    cudaGLMapBufferObject((void**)&m_currentDevice, m_currentVBO_CUDA);
    glBindTexture(GL_TEXTURE_2D, 0);

}


// called by qt whenever the window is resized
void GLFrame::resizeGL(int w, int h)
{
    m_aspect = (GLdouble)w / (GLdouble)h;
    g_scene.view.aspect = m_aspect;
    m_width = w;
    m_height = h;

    glViewport(0, 0, (GLint)w, (GLint)h);
    m_raytracingNeeded = true;
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

    // draw coordinate axes
    if(m_axesVisible)
        drawCoordSys();

    renderScene();
    drawFullScreenQuad();

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
    m_raytracingNeeded = true;
    updateGL();
}

void GLFrame::wheelEvent(QWheelEvent *ev)
{
    // a delta of 2880 is a full turn of the wheel
    if(ev->modifiers() & Qt::ShiftModifier)
        adjustLight(false, false, true, 0, -0.001*ev->delta());
    else
        adjustCam(false, false, true, 0, -0.001*ev->delta());

    m_raytracingNeeded = true;
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
    m_raytracingNeeded = true;
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
            matRot(inc, a, d*5);

            // multiply transformation matrix with eyepoint to get new perspective
            Vec3d eyeg = Vec3d(g_scene.view.eyepoint);
            GLfloat eye[4] = {eyeg[0],eyeg[1],eyeg[2],.0};
            matMult(eye, inc, eye);
            for (int i=0;i<3;i++)
                g_scene.view.eyepoint[i] = eye[i];

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

        // move eyepoint into scrolling direction
        g_scene.view.eyepoint += Vec3d(.0,.0,dy*1000);

        m_fieldOfView *= 1.0+dy;
        if(m_fieldOfView > 160.0)
            m_fieldOfView = 160.0;
    }
    m_raytracingNeeded = true;
}

// notify when the drawMode should be changed
void GLFrame::setRenderMode(int mode)
{
    m_renderMode = (RenderMode)mode;

    if(mode == CPU) {
        // unset VBO
        cudaGraphicsUnregisterResource(m_currentVBO_CUDA);
        glDeleteBuffers(1, &m_currentVBO);
    }
    else if (mode == GPU) {
        // setup VBO
        cudaGLSetGLDevice(0);
        // Create buffer object and register it with CUDA
        glGenBuffers(1, &m_currentVBO);
        glBindBuffer(GL_ARRAY_BUFFER, m_currentVBO);
        unsigned int size = m_width * m_height * 3;//? * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&m_currentVBO_CUDA, m_currentVBO, cudaGraphicsMapFlagsWriteDiscard);

    }

    m_raytracingNeeded = true;
    updateGL();
}

void GLFrame::setAxesVisibility(bool on)
{
    m_axesVisible = on;
    updateGL();
}

void GLFrame::setAntialiasing(bool on)
{
    m_antialiasing = on;
    m_raytracingNeeded = true;
    updateGL();
}

// reset camera to default values
void GLFrame::resetLight()
{
    m_lightDistance = 3.0;

    matIdent(m_lightRot);

    m_raytracingNeeded = true;
    updateGL();
}

// reset camera to default values
void GLFrame::resetCam()
{
    m_distance = 10.0;

    matIdent(m_camRot);

    m_fieldOfView = 15.0;
    m_raytracingNeeded = true;
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

void GLFrame::drawFullScreenQuad()
{

    // setup texture mapping
    glEnable( GL_TEXTURE_2D );

       glMatrixMode(GL_PROJECTION);
       glPushMatrix();
       glLoadIdentity();
       glMatrixMode(GL_MODELVIEW);
       glPushMatrix();
       glLoadIdentity();

       glBindTexture(GL_TEXTURE_2D, m_texHandle);

       glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
       // set to red
       glColor3f(1.0,0.0,.0);

       glBegin(GL_QUADS);
       glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.00f, -1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.00f, -1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.00f,  1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.00f,  1.0f);
       glEnd();

       glMatrixMode(GL_PROJECTION);
       glPopMatrix();
       glMatrixMode(GL_MODELVIEW);
       glPopMatrix();

}

// render scene and save it into a texture
void GLFrame::renderScene()
{
    // only render the scene if there is some intialized raytracer
    if(!m_raytracingNeeded || !m_raytracer || !m_raytracer->m_isFileLoaded)
        return;

    // draw scene
    switch(m_renderMode)
    {
        case CPU:
            m_data = (float*)malloc(sizeof(float) * 3 * m_width * m_height);
            m_raytracer->render(m_data, m_width, m_height);
            loadTexture();
            m_raytracingNeeded = false;
            break;
        case GPU:
            // Map buffer object for writing from CUDA
            cudaGraphicsMapResources(1, &m_currentVBO_CUDA, 0);
            size_t num_bytes;
            cudaGraphicsResourceGetMappedPointer((void**)&m_data,
                                                 &num_bytes,
                                                 m_currentVBO_CUDA);
            // execute rendering on gpu with kernel
            m_raytracer->renderCuda(m_data, m_width, m_height);
            // Unmap buffer object
            cudaGraphicsUnmapResources(1, &m_currentVBO_CUDA, 0);
            // Render from buffer object
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glBindBuffer(GL_ARRAY_BUFFER, m_currentVBO);
            glVertexPointer(3, GL_FLOAT, 0, 0);
            glEnableClientState(GL_VERTEX_ARRAY);
            glDrawArrays(GL_POINTS, 0, m_width * m_height);
            glDisableClientState(GL_VERTEX_ARRAY);
            // Swap buffers
            swapBuffers();
            // indicate done job
            m_raytracingNeeded = false;
            break;
    }
}


void GLFrame::loadScene(const QString &filename)
{
    // create a new raytracer
    delete m_raytracer;
    m_raytracer = new Raytracer(filename.toStdString().c_str(), m_antialiasing);
    // initialize the cuda structure
    m_raytracer->initCuda();
    // and beg for drawing
    m_raytracingNeeded = true;
    updateGL();
}

int GLFrame::frameCounter() const
{
    return m_frameCounter;
}
    
