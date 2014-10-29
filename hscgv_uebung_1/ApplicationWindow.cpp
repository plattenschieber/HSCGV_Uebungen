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
#include "UserParameterDialog.h"
#include "GLFrame.h"

// create an application window
ApplicationWindow::ApplicationWindow()
{
    ui.setupUi(this);

    // set center widget
    m_frame = new GLFrame(this);
    setCentralWidget(m_frame);

    QActionGroup *materialActions = new QActionGroup(this);
    for(int i=0; i<m_frame->numMaterials(); ++i)
    {
        QAction *act = new QAction(m_frame->materialName(i), this);
        act->setCheckable(true);
        m_materialActions.push_back(act);
        materialActions->addAction(act);
        ui.menuMaterial->addAction(act);
        connect(act, SIGNAL(toggled(bool)), this, SLOT(materialChanged()));
    }

    QActionGroup *shaderActions = new QActionGroup(this);
    shaderActions->addAction(ui.actionFixedFunction);
    shaderActions->addAction(ui.actionSimple);
    shaderActions->addAction(ui.actionPhong);
    shaderActions->addAction(ui.actionFreestyle);

    QActionGroup *glActions = new QActionGroup(this);
    glActions->addAction(ui.actionImmediateMode);
    glActions->addAction(ui.actionVertexArrays);
    glActions->addAction(ui.actionVertexBufferObjects);

    QActionGroup *renderModeActions = new QActionGroup(this);
    renderModeActions->addAction(ui.actionWireframe);
    renderModeActions->addAction(ui.actionFlat);
    renderModeActions->addAction(ui.actionSmooth);

    // connect actions
    connect(ui.actionQuit, SIGNAL(triggered()), this, SLOT(fileQuit()));
    connect(ui.actionUserParam, SIGNAL(triggered()), this, SLOT(userParamDialog()));

    connect(ui.actionResetCamera, SIGNAL(triggered()), this, SIGNAL(resetCam()));
    connect(ui.actionResetLight, SIGNAL(triggered()), this, SIGNAL(resetLight()));

    connect(ui.actionAnimate, SIGNAL(toggled(bool)), this, SLOT(animate(bool)));

    connect(ui.actionImmediateMode, SIGNAL(toggled(bool)), SLOT(glModeChanged()));
    connect(ui.actionVertexArrays, SIGNAL(toggled(bool)), SLOT(glModeChanged()));
    connect(ui.actionVertexBufferObjects, SIGNAL(toggled(bool)), SLOT(glModeChanged()));

    connect(ui.actionFixedFunction, SIGNAL(toggled(bool)), SLOT(shaderChanged()));
    connect(ui.actionSimple, SIGNAL(toggled(bool)), SLOT(shaderChanged()));
    connect(ui.actionPhong, SIGNAL(toggled(bool)), SLOT(shaderChanged()));
    connect(ui.actionFreestyle, SIGNAL(toggled(bool)), SLOT(shaderChanged()));

    connect(ui.actionSmooth, SIGNAL(toggled(bool)), SLOT(renderModeChanged()));
    connect(ui.actionFlat, SIGNAL(toggled(bool)), SLOT(renderModeChanged()));
    connect(ui.actionWireframe, SIGNAL(toggled(bool)), SLOT(renderModeChanged()));

    // ----- SIGNALS -----

    // connect signals
    connect(m_frame, SIGNAL(info(QString)), this, SLOT(updateMessage(QString)));

    connect(this, SIGNAL(resetCam()), m_frame, SLOT(resetCam()));
    connect(this, SIGNAL(resetLight()), m_frame, SLOT(resetLight()));

    // create options dialog, but do not show it for now
    m_userParamDialog = new UserParameterDialog(this);
    connect(m_userParamDialog, SIGNAL(userParameter(double)), m_frame, SLOT(setUserParameter(double)));

    connect(ui.actionLoadFile, SIGNAL(triggered()), this, SLOT(loadFile()));
    connect(this, SIGNAL(openFile(const QString&)), m_frame, SLOT(loadModel(const QString&)));

    connect(this, SIGNAL(shader(int)), m_frame, SLOT(setShader(int)));
    connect(this, SIGNAL(glMode(int)), m_frame, SLOT(setGlMode(int)));
    connect(this, SIGNAL(renderMode(int)), m_frame, SLOT(setRenderMode(int)));
    connect(this, SIGNAL(material(int)), m_frame, SLOT(setMaterial(int)));

    connect(ui.actionWireframeOverlay, SIGNAL(toggled(bool)), m_frame, SLOT(setWireframeOverlayVisibility(bool)));
    connect(ui.actionCoordinateAxes, SIGNAL(toggled(bool)), m_frame, SLOT(setAxesVisibility(bool)));
    connect(ui.actionModel, SIGNAL(toggled(bool)), m_frame, SLOT(setModelVisibility(bool)));

    // trigger signals in order to make state known to GLFrame
    m_userParamDialog->restoreState();

    // timer for updating fps display
    QTimer *timer = new QTimer(this);
    timer->setInterval(1000);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateFps()));
    timer->start();

    // timer for continuous rendering
    m_trigger = new QTimer(this);
    m_trigger->setInterval(1); // every ms
    connect(m_trigger, SIGNAL(timeout()), m_frame, SLOT(updateGL()));

    initState();

    if(qApp->argc() > 1)
        loadFile(qApp->argv()[1]);
    else
        loadFile("bunny.ply");
}

void ApplicationWindow::initState() const
{
    // trigger toggles the action's state

    ui.actionCoordinateAxes->setChecked(false);
    ui.actionCoordinateAxes->trigger();
    ui.actionModel->setChecked(false);
    ui.actionModel->trigger();
    ui.actionWireframeOverlay->setChecked(true);
    ui.actionWireframeOverlay->trigger();

    ui.actionAnimate->setChecked(true);
    ui.actionAnimate->trigger();

    ui.actionFixedFunction->trigger();
    ui.actionImmediateMode->trigger();
    ui.actionFlat->trigger();

    m_materialActions[0]->setChecked(false);
    m_materialActions[0]->trigger();
}


// destroy the application window
ApplicationWindow::~ApplicationWindow()
{
    // we don't have to delete anything as this is done automatically for childs
}


// ----- File Menu -----

// file/Options menu method
void ApplicationWindow::userParamDialog()
{
    m_userParamDialog->show();
    m_userParamDialog->raise();
}


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
    m_secondsToDisplay--;
    int frames = m_frame->frameCounter() - oldFrames;
    oldFrames = m_frame->frameCounter();

    if(m_secondsToDisplay >= 0)
        statusBar()->showMessage(m_message);
    else if(frames > 0)
        statusBar()->showMessage(QString::number(frames) + " f/s");
    else
        statusBar()->showMessage("Use mouse buttons and wheel to move camera, hold shift to move light");
}

void ApplicationWindow::loadFile()
{
    QString filename = QFileDialog::getOpenFileName(this,
            tr("Open Model File"), "",
            tr("PLY Models (*.ply);;"
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

void ApplicationWindow::animate(bool on)
{
    if(on)
        m_trigger->start();
    else
        m_trigger->stop();
}


int ApplicationWindow::getRenderMode() const
{
    if(ui.actionWireframe->isChecked())
        return GLFrame::WIRE_FRAME;
    else if(ui.actionSmooth->isChecked())
        return GLFrame::SMOOTH_SHADE;
    else
        return GLFrame::FLAT_SHADE;
}

int ApplicationWindow::getShader() const
{
    if(ui.actionFixedFunction->isChecked())
        return GLFrame::SH_NONE;
    else if(ui.actionSimple->isChecked())
        return GLFrame::SH_SIMPLE;
    else if(ui.actionPhong->isChecked())
        return GLFrame::SH_PHONG;
    else
        return GLFrame::SH_FREESTYLE;
}

int ApplicationWindow::getGlMode() const
{
    if(ui.actionImmediateMode->isChecked())
        return GLFrame::DRAW_IMM;
    else if(ui.actionVertexArrays->isChecked())
        return GLFrame::DRAW_ARRAY;
    else
        return GLFrame::DRAW_VBO;
}

void ApplicationWindow::shaderChanged() const
{
    emit shader(getShader());
}

void ApplicationWindow::glModeChanged() const
{
    emit glMode(getGlMode());
}

void ApplicationWindow::renderModeChanged() const
{
    emit renderMode(getRenderMode());
}

void ApplicationWindow::materialChanged() const
{
    emit material(getMaterial());
}

int ApplicationWindow::getMaterial() const
{
    for(size_t i=0; i<m_materialActions.size(); ++i)
    {
        if(m_materialActions[i]->isChecked())
            return i;
    }

    return 0;
}
