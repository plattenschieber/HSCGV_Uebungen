/** \file
 * \brief OpenSceneGraph in a Qt OpenGL widget
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "qosgviewer.h"

#include <iostream>

QOSGAdapter::QOSGAdapter(QWidget *parent, const QGLWidget* shareWidget, Qt::WindowFlags f)
: QGLWidget(parent, shareWidget, f)
{
    m_gw = new osgViewer::GraphicsWindowEmbedded(0, 0, width(), height());
    m_handleMouseEvents = true;
}

QOSGAdapter::~QOSGAdapter()
{

}

void QOSGAdapter::setHandleMouseEvents(const bool handleMouseEvents)
{
    m_handleMouseEvents = handleMouseEvents;
}

osgViewer::GraphicsWindow* QOSGAdapter::getGraphicsWindow()
{
    return m_gw.get();
}

const osgViewer::GraphicsWindow* QOSGAdapter::getGraphicsWindow() const
{
    return m_gw.get();
}

void QOSGAdapter::resizeGL(const int w, const int h)
{
    m_gw->getEventQueue()->windowResize(0, 0, w, h);
    m_gw->resized(0, 0, w, h);
}

void QOSGAdapter::keyPressEvent(QKeyEvent* event)
{
    m_gw->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol) *(event->text().toAscii().data()));
    emit keyPressed(event);
}

void QOSGAdapter::keyReleaseEvent(QKeyEvent* event)
{
    m_gw->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol) *(event->text().toAscii().data()));
}

void QOSGAdapter::mouseMoveEvent(QMouseEvent* event)
{
    if (m_handleMouseEvents)
    {
        m_gw->getEventQueue()->mouseMotion(event->x(), event->y());
    }
}

void QOSGAdapter::mousePressEvent(QMouseEvent* event)
{
    if (m_handleMouseEvents)
    {
        int button = getOsgMouseButtonCode(event->button());
        m_gw->getEventQueue()->mouseButtonPress(event->x(), event->y(), button);
    }
}

void QOSGAdapter::mouseReleaseEvent(QMouseEvent* event)
{
    if (m_handleMouseEvents)
    {
        int button = getOsgMouseButtonCode(event->button());
        m_gw->getEventQueue()->mouseButtonRelease(event->x(), event->y(), button);
    }
}

void QOSGAdapter::wheelEvent(QWheelEvent *event)
{
    if (m_handleMouseEvents)
    {
        m_gw->getEventQueue()->mouseScroll((event->delta()>0) ?
                osgGA::GUIEventAdapter::SCROLL_DOWN :
                osgGA::GUIEventAdapter::SCROLL_UP);
    }
}

int QOSGAdapter::getOsgMouseButtonCode(const Qt::MouseButton button) const
{
    int result = 0;
    switch (button)
    {
        case Qt::LeftButton:
            result = 1;
            break;
        case Qt::MidButton:
            result = 2;
            break;
        case Qt::RightButton:
            result = 3;
            break;
        case Qt::NoButton:
            result = 0;
            break;
        default:
            result = 0;
            break;
    }
    return result;
}

QSplitscreenViewer::QSplitscreenViewer(QWidget* parent, const QGLWidget* shareWidget, Qt::WindowFlags f)
: QOSGAdapter(parent, shareWidget, f)
{
    setFocusPolicy(Qt::StrongFocus);

    for (int i=0; i<NumViews; ++i)
    {
        m_views[i] = new osgViewer::View;
        m_views[i]->getCamera()->setGraphicsContext(getGraphicsWindow());
        addView(m_views[i]);
    }

    setSplitscreen(false);

    setThreadingModel(osgViewer::Viewer::SingleThreaded);

    setAutoBufferSwap(false);
}

void QSplitscreenViewer::setCameraManipulator(CameraManipulator* manipulator)
{
    for (int i=0; i<NumViews; ++i)
    {
        m_views[i]->setCameraManipulator(manipulator);
    }
}

void QSplitscreenViewer::setCameraManipulator(CameraManipulator* manipulator, const int viewport)
{
    m_views[viewport]->setCameraManipulator(manipulator);
}

void QSplitscreenViewer::setCameraManipulators(CameraManipulator* manipulator1,
        CameraManipulator* manipulator2)
{
    if (NumViews > 0)
    {
        m_views[0]->setCameraManipulator(manipulator1);
    }

    for (int i=1; i<NumViews; ++i)
    {
        m_views[i]->setCameraManipulator(manipulator2);
    }
}

void QSplitscreenViewer::setSceneData(osg::Node* node)
{
    for (int i=0; i<NumViews; ++i)
    {
        m_views[i]->setSceneData(node);
    }
    emit sceneDataChanged();
}

void QSplitscreenViewer::setSplitscreen(const bool splitscreen)
{
    m_splitscreen = splitscreen;

    int widths[NumViews];

    if (m_splitscreen)
    {
        const int w = static_cast<int>(static_cast<float>(width()) / static_cast<float>(NumViews));
        for (int i=0; i<NumViews; ++i)
        {
            widths[i] = w;
        }
    }
    else
    {
        if (NumViews > 0)
        {
            widths[0] = width();
        }

        for (int i=1; i<NumViews; ++i)
        {
            widths[i] = 0;
        }
    }

    const double fovy = splitscreen ? 45.0 : 30.0;
    int currentX = 0;
    for (int i=0; i<NumViews; ++i)
    {
        m_views[i]->getCamera()->setViewport(new osg::Viewport(currentX, 0, widths[i], height()));
        m_views[i]->getCamera()->setProjectionMatrixAsPerspective(fovy, static_cast<double>(widths[i]) / static_cast<double>(height()),
                1.0, 10000.0);
        currentX += widths[i];
    }
}

osg::Camera* QSplitscreenViewer::getCamera(const unsigned int idx) const
{
    return m_views[idx]->getCamera();
}

void QSplitscreenViewer::paintEvent(QPaintEvent *)
{
    QGLWidget::makeCurrent();
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.beginNativePainting();

    // save Qt's OpenGL state
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // handle possible resizing
    m_gw->getEventQueue()->windowResize(0, 0, width(), height());
    m_gw->resized(0, 0, width(), height());

    // render OpenSceneGraph scene
    frame();

    // restore Qt's OpenGL state
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    painter.endNativePainting();
    paintOverlay(&painter);
    painter.end();

    // swap buffers
    if(doubleBuffer())
        swapBuffers();
}
