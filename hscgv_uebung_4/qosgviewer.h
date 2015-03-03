/** \file
 * \brief OpenSceneGraph in a Qt OpenGL widget
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef QOSGVIEWER_H
#define QOSGVIEWER_H

#include <osg/Config>
#include <osg/Version>

#include <osgViewer/CompositeViewer>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>

#include <QGLWidget>
#include <QKeyEvent>
#include <QMouseEvent>

//! OSG Viewer integrated in a Qt Widget, handles events and provides an osg GraphicsWindow.
class QOSGAdapter : public QGLWidget
{
    Q_OBJECT
    signals:
        //! Signal a key press event to the rest of the world.
        void keyPressed(QKeyEvent* event);
    public:
        //! Public constructor.
        QOSGAdapter(QWidget* parent = 0, const QGLWidget* shareWidget = 0, Qt::WindowFlags f = 0);
        //! Public destructor.
        ~QOSGAdapter();

        //! Handle mouse events?
        void setHandleMouseEvents(bool handleMouseEvents);

        //! Access to the low level graphics window.
        osgViewer::GraphicsWindow* getGraphicsWindow();
        //! Access to the low level graphics window (const).
        const osgViewer::GraphicsWindow* getGraphicsWindow() const;
    protected:
        //! Implement Qt's resizeGL function.
        virtual void resizeGL(int w, int h);

        //! Key press event handler.
        virtual void keyPressEvent(QKeyEvent* event);
        //! Key release event handler.
        virtual void keyReleaseEvent(QKeyEvent* event);
        //! Mouse move event handler.
        virtual void mouseMoveEvent(QMouseEvent* event);
        //! Mouse press event handler.
        virtual void mousePressEvent(QMouseEvent* event);
        //! Mouse release event handler.
        virtual void mouseReleaseEvent(QMouseEvent* event);
        //! Mouse wheelevent handler.
        virtual void wheelEvent(QWheelEvent *ev);
    protected:
        //! The low level graphics window wrapped through OpenSceneGraph.
        osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_gw;
        //! Handle mouse events?
        bool m_handleMouseEvents;

        //! Translate Qt mouse button code to OSG mouse button code.
        int getOsgMouseButtonCode(Qt::MouseButton button) const;
};

//! Implement QOSGAdapter, provide a facility to show N views next to each other (default: 2 views).
class QSplitscreenViewer : public QOSGAdapter, public osgViewer::CompositeViewer
{
    Q_OBJECT

    public:

#if OPENSCENEGRAPH_MAJOR_VERSION < 3
        typedef osgGA::MatrixManipulator CameraManipulator;
#else
        typedef osgGA::CameraManipulator CameraManipulator;
#endif

        //! Public constructor.
        QSplitscreenViewer(QWidget* parent = 0, const QGLWidget* shareWidget = 0, Qt::WindowFlags f = 0);

        //! Set camera manipulator for all views.
        void setCameraManipulator(CameraManipulator* manipulator);
        //! Set camera manipulator for the specified view.
        void setCameraManipulator(CameraManipulator* manipulator, const int viewport);
        //! Special case: two views. Set both manipulators with one function call.
        void setCameraManipulators(CameraManipulator* manipulator1,
                CameraManipulator* manipulator2);
        //! Set scene data for all views.
        void setSceneData(osg::Node* node);
        //! Splitscreen? Else: show only the first view.
        void setSplitscreen(bool splitscreen);

        //! Get camera of view \param idx .
        osg::Camera* getCamera(unsigned int idx) const;
    signals:
        //! Signal to the world that the scene data changed.
        void sceneDataChanged();
    protected:
        //! Default: 2 views in splitscreen mode.
        static const int NumViews = 2;
        //! Splitscreen? Else: show only the first view.
        bool m_splitscreen;
        //! An array with pointers to the views.
        osgViewer::View* m_views[NumViews];

        //! Implement Qt's paint event.
        virtual void paintEvent(QPaintEvent *);
        //! reimplement and use QPainter to draw onto the OpenSceneGraph scene
        virtual void paintOverlay(QPainter *) {}
};

#endif // QOSGVIEWER_H
