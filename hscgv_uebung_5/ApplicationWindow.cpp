/**
 * \file
 * \brief Implementation of the ApplicationWindow main window class for our application
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 1 - "Ui, OpenGL!"
 *
 * Created by Christian Vogelgsang <Vogelgsang@informatik.uni-erlangen.de>,
 * changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QPixmap>
#include <QStatusBar>
#include <QToolBar>
#include <QFileDialog>
#include <QTimer>
#include <QDebug>

#include "ApplicationWindow.h"

// create an application window
ApplicationWindow::ApplicationWindow()
{
    ui.setupUi(this);

    // set center widget
    m_frame = new GLFrame(this);
    setCentralWidget(m_frame);

/*    QActionGroup *renderModeActions = new QActionGroup(this);
    renderModeActions->addAction(ui.actionGPU);
    renderModeActions->addAction(ui.actionAntialiasing);*/

    // connect actions
    connect(ui.actionQuit, SIGNAL(triggered()), this, SLOT(fileQuit()));

    connect(ui.actionResetCamera, SIGNAL(triggered()), this, SIGNAL(resetCam()));
    connect(ui.actionResetLight, SIGNAL(triggered()), this, SIGNAL(resetLight()));

    connect(ui.actionAntialiasing, SIGNAL(toggled(bool)), SLOT(antialiasing(bool)));

    connect(ui.actionGPU, SIGNAL(toggled(bool)), m_frame, SLOT(setRenderMode(bool)));

    // ----- SIGNALS -----

    // connect signals
    connect(m_frame, SIGNAL(info(QString)), this, SLOT(updateMessage(QString)));

    connect(this, SIGNAL(resetCam()), m_frame, SLOT(resetCam()));
    connect(this, SIGNAL(resetLight()), m_frame, SLOT(resetLight()));

    connect(ui.actionLoadFile, SIGNAL(triggered()), this, SLOT(loadFile()));
    connect(this, SIGNAL(openFile(const QString&)), m_frame, SLOT(loadScene(const QString&)));

    connect(this, SIGNAL(renderMode(bool)), m_frame, SLOT(setRenderMode(bool)));

    // timer for updating fps display
    QTimer *timer = new QTimer(this);
    timer->setInterval(1000);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateFps()));
    timer->start();

    // timer for continuous rendering
    m_trigger = new QTimer(this);
    connect(m_trigger, SIGNAL(timeout()), m_frame, SLOT(updateGL()));
    m_trigger->start();

    // load standard scene
//    if(qApp->argc() > 1)
//        loadFile(qApp->argv()[1]);
//    else
//        loadFile("REF1.data");
}

// destroy the application window
ApplicationWindow::~ApplicationWindow()
{
    // we don't have to delete anything as this is done automatically for childs
}


// ----- File Menu -----

// file/Quit menu method
void ApplicationWindow::fileQuit()
{
    qApp->quit();
}

void ApplicationWindow::updateMessage(const QString& message)
{
    m_secondsToDisplay = 6;
    m_message = message;
    statusBar()->showMessage(message);
}


void ApplicationWindow::updateFps()
{
    static int oldFrames = 0;
    int frames = m_frame->frameCounter() - oldFrames;
    oldFrames = m_frame->frameCounter();

    statusBar()->showMessage(QString::number(frames) + " f/s");
}

void ApplicationWindow::loadFile()
{
    QString filename = QFileDialog::getOpenFileName(this,
            tr("Open Model File"), "",
            tr("Raytracing Scenes (*.data);;"
                "All Files (*)"));

    if(!filename.isEmpty())
    {
        loadFile(filename);
    }
}

void ApplicationWindow::loadFile(QString filename)
{
        QString msg;
        msg.sprintf("Loading %s...", filename.toLocal8Bit().constData());
        updateMessage(msg);
        emit openFile(filename);
}

void ApplicationWindow::antialiasing(bool on)
{
    m_frame->m_antialiasing = on;
}


