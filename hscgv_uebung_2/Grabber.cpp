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
, m_animationPhase(NO_ANIMATION)
{
   m_sceneGraph = initSceneGraph();
   idleSensor = new SoIdleSensor;
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

   m_gameboard->setPiece(m_pickedUpTile->getChild(0));
   m_pickedUpTile->getChild(0)->unref();
   m_pickedUpTile->removeChild(0);
   // remember to drop down a piece
//   m_mode = SET_PIECE;
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

    // get the piece from the gameboard and attach it to the grabber (beneath finger)
    m_pickedUpTile->addChild(m_gameboard->getPiece());

   // remember to pick up a piece
//   m_mode = GET_PIECE;
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
//* Grabber idleSensor Callback
//**********************************************************
//* This routine is called whenever the CPU is idle.
//* We determine the state of the grabber and
//* let the grabber accordingly move or not
//*
//**********************************************************
void
Grabber::myIdleCB(void *userData, SoSensor *sensor)
{


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
   transGrabber->translation.setValue(12.,12.,0.);
   grabber->addChild(transGrabber);


   // ------------------- shoulder begin --------------------
   SoSeparator *shoulder = new SoSeparator;
   grabber->addChild(shoulder);
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

   // define colors for the faces
   SoMaterial *shoulderMaterials = new SoMaterial;
   shoulderMaterials->diffuseColor.setValues(0, 8, shoulderColors);
   shoulder->addChild(shoulderMaterials);
   // bind defined material to faces
   SoMaterialBinding *shoulderMaterialBinding = new SoMaterialBinding;
   shoulderMaterialBinding->value = SoMaterialBinding::PER_FACE;
   shoulder->addChild(shoulderMaterialBinding);
   // define coordinates for vertices
   SoCoordinate3 *shoulderCoords = new SoCoordinate3;
   shoulderCoords->point.setValues(0, 12, shoulderVertices);
   shoulder->addChild(shoulderCoords);
   // scale shoulder
   SoTransform *shoulderScale = new SoTransform;
   shoulderScale->scaleFactor.setValue(.2,.2,.4);
   shoulder->addChild(shoulderScale);
   // define the IndexedFaceSet, with indices into the vertices
   SoIndexedFaceSet *shoulderFaceSet = new SoIndexedFaceSet;
   shoulderFaceSet->coordIndex.setValues(0, 44, shoulderIndices);
   shoulder->addChild(shoulderFaceSet);
   // --------------------- shoulder end ---------------------


   //  ------------------- the moving part of the grabber --------------------
   SoSeparator *movingGrabber = new SoSeparator;
   grabber->addChild(movingGrabber);

   // build arm with relative position to shoulder and size
   SoGroup *armGroup = new SoGroup;
   movingGrabber->addChild(armGroup);
   SoTransform *armTrafo = new SoTransform;
   armTrafo->translation.setValue(.0,.0,4.0);
   armTrafo->rotation.setValue(SbVec3f(0,0,1), 3*M_PI_4);
   armGroup->addChild(armTrafo);
   SoCube *arm = new SoCube;
   arm->width = arm->height = 1.5;
   arm->depth = 5;
   armGroup->addChild(arm);

   // build elbow
   SoGroup *elbowGroup = new SoGroup;
   movingGrabber->addChild(elbowGroup);
   SoTransform *elbowTrafo = new SoTransform;
   // translate elbow to top of shoulder + its own half height (=radius)
   // [all heigths of any SoObjects are measured from their midpoints]
   elbowTrafo->translation.setValue(.0,.0,arm->depth.getValue()/2+.75);
   elbowTrafo->rotation.setValue(SbVec3f(0,1,0), M_PI_4);
   elbowGroup->addChild(elbowTrafo);
   SoCylinder *elbow = new SoCylinder;
   elbow->radius = 1.;
   elbow->height = 1.5;
   elbowGroup->addChild(elbow);

   // build forarm consisting of three parts
   // firstly build a basic part rotated by 90 degrees around the z-axis
   SoGroup *forearmGroup = new SoGroup;
   movingGrabber->addChild(forearmGroup);
   SoSeparator *forearm = new SoSeparator;
   SoTransform *forearmTrafo = new SoTransform;
   forearmTrafo->rotation.setValue(SbVec3f(0,0,1), M_PI/2.);
   forearm->addChild(forearmTrafo);
   SoCylinder *forearm1 = new SoCylinder;
   forearm1->radius = .75;
   forearm1->height = 3.3;
   forearm->addChild(forearm1);
   // then add this part three times in different scales
   // part 1 -----------------
   SoTransform *forearmTrafo1 = new SoTransform;
   // move the forearm to the edge of the elbow
   forearmTrafo1->translation.setValue(elbow->radius.getValue()+3.3/2.,.0,.0);
   forearmGroup->addChild(forearmTrafo1);
   forearmGroup->addChild(forearm);
   // part 2 -----------------
   SoTransform *forearmTrafo2 = new SoTransform;
   // move the forearm to the end of the last part
   forearmTrafo2->translation.setValue(3.3,.0,.0);
   // each part of the forearm gets smaller
   forearmTrafo2->scaleFactor.setValue(1.,.5,.5);
   forearmGroup->addChild(forearmTrafo2);
   forearmGroup->addChild(forearm);
   // part 3 -----------------
   SoTransform *forearmTrafo3 = new SoTransform;
   // move the forearm to the end of the last part
   forearmTrafo3->translation.setValue(3.3,.0,0.0);
   // each part of the forearm gets smaller
   forearmTrafo3->scaleFactor.setValue(1.,.5,.5);
   forearmGroup->addChild(forearmTrafo3);
   forearmGroup->addChild(forearm);

   // build wrist
   SoGroup *wristGroup = new SoGroup;
   movingGrabber->addChild(wristGroup);
   SoTransform *wristTrafo = new SoTransform;
   wristTrafo->translation.setValue(3.3/2+.5,.0,0.0);
   // here we need to scale back to 1 (revert the two .5 scales before)
   wristTrafo->scaleFactor.setValue(1.,4.,4.);
   wristTrafo->rotation.setValue(SbVec3f(0,0,0), 0);
   wristGroup->addChild(wristTrafo);
   SoSphere *wrist = new SoSphere;
   wrist->radius = .5;
   wristGroup->addChild(wrist);

   // build hand
   SoGroup *handGroup = new SoGroup;
   movingGrabber->addChild(handGroup);
   SoTransform *handTrafo = new SoTransform;
   // go down in z-axis direction
   handTrafo->translation.setValue(.0,.0,-2*wrist->radius.getValue());
   handTrafo->rotation.setValue(SbVec3f(1,0,0), -M_PI_2);
   handGroup->addChild(handTrafo);
   SoCone *hand = new SoCone;
   hand->bottomRadius = .5;
   hand->height = 1.;
   handGroup->addChild(hand);

   // and last but not least, the finger
   SoGroup *fingerGroup = new SoGroup;
   movingGrabber->addChild(fingerGroup);
   SoTransform *fingerTrafo = new SoTransform;
   fingerTrafo->translation.setValue(.0,hand->height.getValue()/2,.0);
   fingerTrafo->rotation.setValue(SbVec3f(0,0,0), 0);
   fingerGroup->addChild(fingerTrafo);
   SoCylinder *finger = new SoCylinder;
   finger->radius = .5;
   finger->height = .25;
   fingerGroup->addChild(finger);

   // and lust but not leaster ;) - we need a place for the picked up tile
   m_pickedUpTile = new SoGroup;
   movingGrabber->addChild(m_pickedUpTile);

   // build remaining parts of grabber
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
