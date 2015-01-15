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
#include <Inventor/nodes/SoMaterial.h>

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
   // move the grabber to a nice place
   SoTransform *transGrabber = new SoTransform;
   transGrabber->translation.setValue(13.,13.,0.);
   grabber->addChild(transGrabber);


   // positions of all of the vertices
   static float shoulderVertices[12][3] =
   {
       { 8.0000,  0.0000,  0.0000},  // P1 - lower part of shoulder
       { 4.0000,  7.0000,  0.0000},  // P2  |
       {-4.0000,  7.0000,  0.0000},  // P3  |
       {-8.0000,  0.0000,  0.0000},  // P4  |
       {-4.0000, -7.0000,  0.0000},  // P5  |
       { 4.0000, -7.0000,  0.0000},  // P6  |
       { 8.0000,  0.0000,  4.0000},  // P7 - upper part of shoulder
       { 4.0000,  7.0000,  4.0000},  // P8  |
       {-4.0000,  7.0000,  4.0000},  // P9  |
       {-8.0000,  0.0000,  4.0000},  // P10 |
       {-4.0000, -7.0000,  4.0000},  // P11 |
       { 4.0000, -7.0000,  4.0000},  // P12 |
   };

   // connectivity information:
   // 2 faces with 6 vertices each (top and bottom faces)
   // 6 faces with 4 vertices each (side faces)
   // (plus the end-of-face indicator for each face)
   static int shoulderIndices[44] =
   {
       1, 2, 3,  4,  5, SO_END_FACE_INDEX,    // bottom face
       6, 7, 8, 9, 10, 11, SO_END_FACE_INDEX, // top face
       0, 1,  7,  6, SO_END_FACE_INDEX,       // side faces
       1, 2,  8,  7, SO_END_FACE_INDEX,       // |
       2, 3,  9,  8, SO_END_FACE_INDEX,       // |
       3, 4, 10,  9, SO_END_FACE_INDEX,       // |
       4, 5, 11, 10, SO_END_FACE_INDEX,       // |
       5, 0,  6, 11, SO_END_FACE_INDEX,       // |
   };

   // colors for the 8 faces
   static float shoulderColors[8][3] =
   {
       {1.0,  .0,  0}, {1.0,  .0,  0},                 // upper and lower face colours
       {1.0,  .0,  0}, {1.0,  .0,  0}, {1.0,  .0,  0}, // side face colours
       {1.0,  .0,  0}, {1.0,  .0,  0}, {1.0,  .0,  0}, // |
   };

   // Define colors for the faces
   SoMaterial *myMaterials = new SoMaterial;
   myMaterials->diffuseColor.setValues(0, 8, shoulderColors);
   grabber->addChild(myMaterials);
   SoMaterialBinding *myMaterialBinding = new SoMaterialBinding;
   myMaterialBinding->value = SoMaterialBinding::PER_FACE;
   grabber->addChild(myMaterialBinding);

   // Define coordinates for vertices

   // Define coordinates for vertices
   SoCoordinate3 *myCoords = new SoCoordinate3;
   myCoords->point.setValues(0, 12, shoulderVertices);
   grabber->addChild(myCoords);

   // Define the IndexedFaceSet, with indices into
   // the vertices:
   SoIndexedFaceSet *myFaceSet = new SoIndexedFaceSet;
   myFaceSet->coordIndex.setValues(0, 44, shoulderIndices);
   grabber->addChild(myFaceSet);


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
