/** \file
 * \brief Control the simulation and process user input
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 * and Stefan Zellmann <zellmans@uni-koeln.de>
 */

#ifndef APPWIN_H
#define APPWIN_H

#include <osg/ref_ptr>

#include <QKeyEvent>
#include <QMainWindow>
#include <QTimer>

#include <vector>

#include "lbm.h"
#include "slice.h"
#include "slicevisualisation.h"
#include "linevisualisation.h"

class OverlayViewer;
class HelpDialog;

//! process user input and control simuluation.
class ApplicationWindow : public QObject
{
    Q_OBJECT
    public:
        //! Public constructor.
        ApplicationWindow();
        //! Public destructor.
        ~ApplicationWindow();

    private slots:
        //! Handle Qt key events.
        void handleKeyEvent(QKeyEvent* event);
        //! Open a help dialog about keyboard controls for the players.
        void openHelpDialog();
        //! Start a new simulation
        void start(int initial);
        //! Draw frames, update simulation
        void update();
    private:
        //! The main window.
        QMainWindow* m_mainWindow;
        //! A viewer for OpenSceneGraph scenes
        OverlayViewer* m_viewer;
        //! Qt action: exit program.
        QAction* m_exitAction;

        // Help menu actions.
        QAction* m_controlsAction;

        //! Help dialog, show keyboard controls.
        HelpDialog* m_helpDialog;

        //! Init Qt menu.
        void initMenu();

        //! Timer for frame rendering.
        QTimer m_timer;

        //! LBM simulation
        LBMD3Q19 *m_lbm;

        //! root node of visualisation
        osg::ref_ptr<osg::Group> m_root;

        //! line visualisation
        osg::ref_ptr<LineVisualisation> m_lines;

        //! slice visualisation
        osg::ref_ptr<SliceVisualisation> m_slices;

        //! current axis for moving slice
        int m_currentAxis;

        //! current value (velocity, density)
        Slice::DataValue m_value;

        //! flag to inhibit simulation updates
        bool m_paused;

        //! flag for simulation on GPU
        bool m_useGpu;
};

#endif
