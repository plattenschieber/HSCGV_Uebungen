/* *************** Programmierpraktikum Computergrafik (CGP) ***************
 * Aufgabe 2 - "Der Greifer"
 *
 * Created by Kai Hormann <Hormann@informatik.uni-erlangen.de>
 * and        Christof Rezk-Salama <Rezk@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: Grabber.cpp
 * Description: 
 *   Implementation of the Grabber robot arm
 */

#include <cmath>
#include <iostream>

// Inventor includes
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoCylinder.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoSphere.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/sensors/SoIdleSensor.h>
#include <Inventor/engines/SoCompose.h>
#include <Inventor/engines/SoCalculator.h>

// own includes
#include "Grabber.h"
#include "Gameboard.h"


//**********************************************************
//* Grabber Constructor
//**********************************************************
//* This method is called when the class is constructed using
//* the new command. No arguments 
//* 
//**********************************************************
Grabber::Grabber()
: m_sceneGraph(NULL)
, m_gameboard(NULL)
, m_mode(INACTIVE)
{
   m_sceneGraph = initSceneGraph();
}


//**********************************************************
//* Grabber Destructor
//**********************************************************
//* This method is called when the class is destroyed using
//* the delete command. Do the clean up stuff in here!
//* 
//**********************************************************
Grabber::~Grabber()
{
   // unreferencing of the scenegraph
   if (m_sceneGraph != NULL)
      m_sceneGraph->unref();
}



//**********************************************************
//* Grabber setPiece
//**********************************************************
//* This method is called by the gameboard to tell the
//* grabber, that he should move his arm to the speified
//* position and drop down the piece
//* 
//**********************************************************
void
Grabber::setPiece(SbVec3f position)
{
   // Schedule the idle sensor for the animation!
   // During the animation Gameboard::setPiece is called.

   // remember to drop down a piece
   m_mode = SET_PIECE;
}

//**********************************************************
//* Grabber getPiece
//**********************************************************
//* This method is called by the gameboard to tell the
//* grabber, that he should move his arm to the speified
//* position and pick up a piece
//* 
//**********************************************************
void
Grabber::getPiece(SbVec3f position)
{
   // Schedule the idle sensor for the animation!
   // During the animation Gameboard::getPiece is called to
   // receive the piece's geometry and add it into our scenegraph

   // remember to pick up a piece
   m_mode = GET_PIECE;
}

//**********************************************************
//* Grabber attachGameboard
//**********************************************************
//* This method is called to attach the grabber to a
//* gameboard.
//* 
//**********************************************************
void
Grabber::attachGameboard(Gameboard *gameboard)
{
   m_gameboard = gameboard;
}




//**********************************************************
//* Grabber initSceneGraph
//**********************************************************
//* build the complete SceneGraph of the grabber
//* 
//**********************************************************
SoSeparator *
Grabber::initSceneGraph()
{
   // Create the grabber
   SoSeparator *grabber = new SoSeparator;

   return grabber;
}

//**********************************************************
//* Grabber isWaiting
//**********************************************************
//* This function can be used to determine wether the
//* grabber is idle or working.
//* 
//**********************************************************
bool 
Grabber::isWaiting()
{
   return (m_mode == INACTIVE);
}
