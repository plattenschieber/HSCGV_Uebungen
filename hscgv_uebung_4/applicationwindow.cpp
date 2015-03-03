/** \file
 * \brief Main application window
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "applicationwindow.h"
#include "helpdialog.h"
#include "overlayviewer.h"
#include "slice.h"
#include "lbm.h"
#include "slicevisualisation.h"
#include "linevisualisation.h"
#include "coordinatebox.h"
#include "clock.h"

#include <osgGA/TrackballManipulator>


#include <QMenuBar>
#include <QApplication>
#include <QDebug>

#include <iostream>

ApplicationWindow::ApplicationWindow()
: m_lbm(NULL)
, m_currentAxis(0)
, m_value(Slice::Density)
, m_paused(false)
, m_useGpu(true)
{
    m_mainWindow = new QMainWindow();
    // Reference counted by Qt if parent is set.
    m_helpDialog = new HelpDialog(m_mainWindow);

    initMenu();

    m_viewer = new OverlayViewer;
    m_viewer->setSplitscreen(false);
    m_viewer->setHandleMouseEvents(true);
    m_mainWindow->setCentralWidget(m_viewer);
    m_mainWindow->resize(500, 500);
    m_mainWindow->show();

    // start simulation with initial state no. 1
    start(1);

    connect(m_viewer, SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(handleKeyEvent(QKeyEvent*)));
    connect(&m_timer, SIGNAL(timeout()), this, SLOT(update()));
    m_timer.start(10);
}

ApplicationWindow::~ApplicationWindow()
{
}

void ApplicationWindow::start(int initial)
{
    m_viewer->setSceneData(NULL);
    delete m_lbm;

    int s = 32;

    switch(initial)
    {
        case 4:
            s *= 2;
        case 3:
            s *= 2;
        case 2:
            s *= 2;
        case 1:
            {
                int w = s, h = s+1, d = s; // less cache thrashing for non power-of-2 sizes
                m_lbm = new LBMD3Q19(w, h, d);

                for(int j=h/10; j<h*9/20; ++j)
                {
                    for(int i=w/10; i<w*9/10; ++i)
                    {
                        m_lbm->setVelocity(i, j, d/4, LBMD3Q19::Vector(0., 0., 0.10));
                        m_lbm->setNoSlip(i, j, d*3/4);
                    }
                }
                m_lbm->setOmega(1.95);
                break;
            }
        default:
        case 0:
            {
                m_lbm = new LBMD3Q19(s, s+1, s); // less cache thrashing for non power-of-2 sizes
                m_lbm->setOmega(1.8);
                break;
            }
    }

    m_lbm->apply();
    m_lbm->useCuda(m_useGpu);

    // scene hierarchy
    m_root = new osg::Group;

    // bounding box around simulation domain
    m_root->addChild(new CoordinateBox(m_lbm->getDimension(0), m_lbm->getDimension(1), m_lbm->getDimension(2)));

    // lines and slices
    m_lines = new LineVisualisation(m_lbm);
    m_root->addChild(m_lines);
    m_slices = new SliceVisualisation(m_lbm);
    m_root->addChild(m_slices);
    m_viewer->setSceneData(m_root);

    osgGA::TrackballManipulator* manipulator = new osgGA::TrackballManipulator;
    m_viewer->setCameraManipulator(manipulator);
}

void ApplicationWindow::update()
{
    const int Steps = 10;

    double mlups = 0.;
    if(!m_paused)
    {
        double start = Clock::now();
        for(int i=0; i<Steps; ++i)
        {
            m_lbm->step();
        }

        m_lbm->sync(); // make time measurement reliable
        double stop = Clock::now();
        mlups = 1e-6*Steps*m_lbm->getDimension(0)*m_lbm->getDimension(1)*m_lbm->getDimension(2)/(stop-start);
    }
    m_lbm->analyze();

    double rhoMin = m_lbm->minDensity();
    double rhoMax = m_lbm->maxDensity();
    double vMax = m_lbm->maxVelocity();

    QString msg;
    msg.sprintf("step: %d   rho: %5.3f - %5.3f   v: %5f   Mlup/s:%6.1f", m_lbm->getStep(), rhoMin, rhoMax, vMax, mlups);

    m_viewer->showInfo(msg);

    m_lines->update();
    m_slices->update(m_value);

    m_viewer->repaint();
}

void ApplicationWindow::openHelpDialog()
{
    m_helpDialog->show();
    m_helpDialog->raise();
}

void ApplicationWindow::initMenu()
{
    QMenuBar* menuBar = m_mainWindow->menuBar();

    // File menu.
    QMenu* fileMenu = menuBar->addMenu(tr("&File"));

    m_exitAction = new QAction(tr("&Quit"), m_mainWindow);
    fileMenu->addAction(m_exitAction);
    connect(m_exitAction, SIGNAL(triggered()), m_mainWindow, SLOT(close()));

    // Help menu.
    QMenu* helpMenu = menuBar->addMenu(tr("&Help"));

    m_controlsAction = new QAction(tr("&Keyboard Controls"), m_mainWindow);
    helpMenu->addAction(m_controlsAction);
    connect(m_controlsAction, SIGNAL(triggered()), this, SLOT(openHelpDialog()));
}

void ApplicationWindow::handleKeyEvent(QKeyEvent* event)
{
    switch (event->key())
    {
        case Qt::Key_Left:
            m_currentAxis = (m_currentAxis+2)%3;
            break;
        case Qt::Key_Right:
            m_currentAxis = (m_currentAxis+1)%3;
            break;
        case Qt::Key_Up:
            m_slices->setSlice(m_currentAxis, m_slices->getSlice(m_currentAxis)+1);
            break;
        case Qt::Key_Down:
            m_slices->setSlice(m_currentAxis, m_slices->getSlice(m_currentAxis)-1);
            break;

        case Qt::Key_V:
            m_value = Slice::Velocity;
            break;
        case Qt::Key_D:
            m_value = Slice::Density;
            break;

        case Qt::Key_1:
            start(1);
            break;
        case Qt::Key_2:
            start(2);
            break;
        case Qt::Key_3:
            start(3);
            break;
        case Qt::Key_4:
            start(4);
            break;
        case Qt::Key_5:
            start(5);
            break;
        case Qt::Key_6:
            start(6);
            break;
        case Qt::Key_7:
            start(7);
            break;
        case Qt::Key_8:
            start(8);
            break;
        case Qt::Key_9:
            start(9);
            break;
        case Qt::Key_0:
            start(0);
            break;

        case Qt::Key_P:
            m_paused = !m_paused;
            break;
        case Qt::Key_Period:
            m_lbm->step();
            break;
        case Qt::Key_C:
            m_useGpu = false;
            m_lbm->useCuda(false);
            break;
        case Qt::Key_G:
            m_useGpu = true;
            m_lbm->useCuda(true);
            break;

        case Qt::Key_L:
            if(m_root->containsNode(m_lines))
                m_root->removeChild(m_lines);
            else
                m_root->addChild(m_lines);
            break;

        case Qt::Key_S:
            if(m_root->containsNode(m_slices))
                m_root->removeChild(m_slices);
            else
                m_root->addChild(m_slices);
            break;

        case Qt::Key_Question:
        case Qt::Key_F1:
            openHelpDialog();
            break;
    }
}
