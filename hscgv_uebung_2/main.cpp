/* *************** Programmierpraktikum Computergrafik (CGP) ***************
 * Aufgabe 2 - "Der Greifer"
 *
 * Created by Kai Hormann <Hormann@informatik.uni-erlangen.de>
 * and        Christof Rezk-Salama <Rezk@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: main.cpp
 * Description: 
 *   SoQt initialisation and main loop
 */

// Qt includes
#include <qwidget.h>

// Inventor includes
#include <Inventor/SoDB.h>
#include <Inventor/Qt/SoQt.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/actions/SoBoxHighlightRenderAction.h>

// own includes
#include "Grabber.h"
#include "Gameboard.h"

int
main(int argc, char *argv[])
{
   // initialize SoQt and create a main window
   QWidget *mainwin = SoQt::init(argc, argv, argv[0]);
   if( !mainwin )
      return 1;

   // initialize scene graph database
   SoDB::init();

   // create the root scenegraph
   SoSeparator *root = new SoSeparator;

   // insert the subgraph of the gameboard
   Gameboard *board = new Gameboard();
   root->addChild(board->getSceneGraph());

   // insert the own subgraph, that draws the grabber
   Grabber   *grabber = new Grabber();
   root->addChild(grabber->getSceneGraph());

   // connect grabber and gameboard
   grabber->attachGameboard(board);
   board->attachGrabber(grabber);

   // initalize the viewer window  
   SoQtExaminerViewer *viewer = new SoQtExaminerViewer(mainwin);
   viewer->setSceneGraph(root);
   viewer->setBackgroundColor(SbColor(0.3, 0.3, 1.0));
   viewer->setTitle("Grabber");
   viewer->show();

   // set the highlight render action (the selected box gets a wireframe around it, to indicate selection)
   SoBoxHighlightRenderAction *renderAct = new SoBoxHighlightRenderAction;
   viewer->setGLRenderAction(renderAct);

   // request automatic redraw when selection changes
   viewer->redrawOnSelectionChange( (SoSelection *) board->getSceneGraph() );

   // display the window and enter the main loop
   SoQt::show(mainwin);
   SoQt::mainLoop();

   delete renderAct;
   return 0;
}
