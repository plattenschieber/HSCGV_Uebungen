/* *************** Programmierpraktikum Computergrafik (CGP) ***************
 * Aufgabe 2 - "Der Greifer"
 *
 * Created by Kai Hormann <Hormann@informatik.uni-erlangen.de>
 * and        Christof Rezk-Salama <Rezk@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: Gameboard.cpp
 * Description: 
 *   Implementation of the "intelligent" gameboard
 */

// Inventor includes

#include <Inventor/SoPath.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoSphere.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoCylinder.h>
#include <iostream>

// own includes
#include "Grabber.h"
#include "Gameboard.h"

//**********************************************************
//* Gameboard Selection Callback
//**********************************************************
//* This routine is called when an object gets selected.
//* We determine which object was selected, and let the
//* gamelogic handle everything else.
//*
//**********************************************************
void
Gameboard::mySelectionCB(void *userData, SoPath *selectionPath)
{
   // we need a pointer to our gameboard
   Gameboard *myGameboard = static_cast<Gameboard *> (userData);
   // convinience, since we don't want to cast in each loop
   SoSelection *mySceneGraph = static_cast<SoSelection *> (myGameboard->m_sceneGraph);
   int index = 0;
   // the square (including tile and square) is the second node on the path
   // compare it to each child of the SoSelection root
   while (selectionPath->getNode(1) != mySceneGraph->getChild(index)) index++;

   // let the game logic check the state and handle all actions
   myGameboard->selectPiece(index);
}


//**********************************************************
//* Gameboard Constructor
//**********************************************************
//* This method is called when the class is constructed using
//* the new command. As argument a VALID instance of class
//* Grabber must be assigned
//* 
//**********************************************************
Gameboard::Gameboard()
: m_sceneGraph(NULL) // scene graph is constructed below
, m_fromSquare(-1) // no square selected
, m_toSquare(-1) // no square selected
, m_state(NO_PIECE_PICKED) // no piece is currently picked
, m_grabber(NULL)
{
   //initialize the gameboard
   initGameboard();
}

//**********************************************************
//* Gameboard Destructor
//**********************************************************
//* This method is called when the class is deleted using
//* the delete command. Do the clean up stuff in here!
//* 
//**********************************************************
Gameboard::~Gameboard()
{
   // unreferencing of the scenegraph
   if (m_sceneGraph != NULL)
      m_sceneGraph->unref();

}

//**********************************************************
//* Gameboard initGameboard
//**********************************************************
//* This method initalized the gameboard internal structures
//* especially the m_squares array to the initial positions
//* of the pieces
//* 
//**********************************************************
void
Gameboard::initGameboard()
{
   // init the squares of the gamebord
   int i,j, index = 0;

   // for all squares
   for(i = 0; i < 7; i++) {
      for(j = 0; j < 7; j++) {

         if (((i < 2) || (i > 4)) && ((j < 2) || (j > 4))) {
            // the squares at the corners of the board are not valid
            m_squares[index] = INVALID_FIELD;
         } else {
            // all other squares are occupied by a piece
            m_squares[index] = OCCUPIED_FIELD; 
         }

         index++;
      }    
   }
   // the middle square (index 24) is empty
   m_squares[24] = EMPTY_FIELD;

   // now use the m_squares array to init the scenegraph
   initSceneGraph();
}

//**********************************************************
//* Gameboard attachGrabber
//**********************************************************
//* This method is called to attach a grabber to the
//* gameboard.
//* 
//**********************************************************
void
Gameboard::attachGrabber(Grabber *grabber)
{
   m_grabber = grabber;
}

//**********************************************************
//* Gameboard initSceneGraph
//**********************************************************
//* This method initalized the scenegraph of the gameboard
//* 
//**********************************************************
void
Gameboard::initSceneGraph()
{
   if (m_sceneGraph != NULL) // this should never happen! 
      m_sceneGraph->unref();  

   //**************************************
   // init the scenegraph of the gameboard
   //**************************************

   // start with a selection node that automatically
   // handles picking events
   m_sceneGraph = new SoSelection;

   // we want to keep the gameboard scene graph during all our life
   m_sceneGraph->ref();

   m_sceneGraph->addSelectionCallback(mySelectionCB, this);

   // we need only four objects: a cube and a sphere for our board and their materials
   SoSphere *sphere = new SoSphere;
   SoCube *cube = new SoCube;
   SoMaterial *cubeBlackMaterial = new SoMaterial;
   SoMaterial *cubeWhiteMaterial = new SoMaterial;
   SoMaterial *sphereMaterial = new SoMaterial;

   // set sphere material to shiny silver
   sphereMaterial->ambientColor.setValue(.2, .9, .2);
   sphereMaterial->diffuseColor.setValue(.6, .6, .6);
   sphereMaterial->specularColor.setValue(.5, .5, .5);
   sphereMaterial->shininess = .5;

   // set black and white cube material
   cubeBlackMaterial->diffuseColor.setValue(0.,0.,0.);
   cubeWhiteMaterial->diffuseColor.setValue(1.,1.,1.);

   // create gamboard and tiles (spheres)
   for (int i=0; i<7; i++) {
       for (int j=0; j<7; j++) {
           // each square needs a seperator and two transformer for cube and sphere
           SoSeparator *currentSep = new SoSeparator;
           SoTransform *cubeTransform = new SoTransform;
           SoTransform *sphereTransform = new SoTransform;

           // the cube and sphere are better hosted in a group
           SoGroup *cubeGroup = new SoGroup;
           SoGroup *sphereGroup = new SoGroup;

           // add our seperation node to the gameboard - represents the square with the tile on it
           // in the case of an invalid field it's just empty
           m_sceneGraph->addChild(currentSep);

           // create the field and tiles
           if (m_squares[i*7+j] != INVALID_FIELD) {
               ///
               /// CUBE
               ///
               // setup the cube group
               currentSep->addChild(cubeGroup);
               // translate cube
               cubeTransform->translation.setValue(j*2.0, i*2.0, .0);
               // paint all even squares white
               if (((i*7+j)%2) == 1)
                   cubeGroup->addChild(cubeWhiteMaterial);
               else // and odd squares black
                   cubeGroup->addChild(cubeBlackMaterial);
               // add all nodes
               cubeGroup->addChild(cubeTransform);
               cubeGroup->addChild(cube);

               ///
               /// SPHERE
               ///
               // add sphere only in case it's not the middle square
               if (m_squares[i*7+j] == OCCUPIED_FIELD) {
                   // setup the sphere group
                   currentSep->addChild(sphereGroup);
                   // resize sphere and place it over the cube (state machine)
                   sphereTransform->scaleFactor.setValue(.5,.5,.5);
                   sphereTransform->translation.setValue(.0,.0,1.5);
                   sphereGroup->addChild(sphereMaterial);
                   sphereGroup->addChild(sphereTransform);
                   sphereGroup->addChild(sphere);
               }
           }
       }
   }
}


//**********************************************************
//* Gameboard setPiece()
//**********************************************************
//* This method inserts the given geometry node of the piece
//* to the current square (whose index is kept in m_toSquare)
//**********************************************************
void
Gameboard::setPiece(SoNode *piece)
{
   // insert the piece
   insertPiece(m_toSquare, piece);
}

//**********************************************************
//* Gameboard getPiece()
//**********************************************************
//* This method removes a piece from the current square (whose
//* index is kept in m_fromSquare) and returns the Scenegraph
//* node of the piece
//**********************************************************
SoNode *
Gameboard::getPiece()
{
   // return the piece scenegraph
   return removePiece(m_fromSquare);
}



//**********************************************************
//* Gameboard selectPiece
//**********************************************************
//* This method checks wether the selection of the square
//* is valid or not and invokes the grabber to get/set the
//* piece to the specified location 
//* 
//**********************************************************
void
Gameboard::selectPiece(int selected)
{
   // check if the grabber is ready 
   if (!m_grabber->isWaiting()) {
      return;
   }


   // The grabber already has a piece
   // and wants to drop it
   //-----------------------------------
   if (m_state == PIECE_PICKED) {

      // an empty square must be selected
      if (m_squares[selected] == EMPTY_FIELD) {
         m_toSquare = selected;

         // EVALUATE THE GAME RULES
         // to check if it's a valid move ...
         int PieceToRemove = evaluateRules();

         if (PieceToRemove > 0) { //  ... if this is the case

            // remove the inbetween piece;
            //-----------------------------------
            // remove the pieces jumped over (and lower their reference count
            // as it was increased in removePiece)
            removePiece(PieceToRemove)->unref();

            // put down the piece;
            //-----------------------------------
            // store the current square
            m_toSquare = selected;
            // remember a piece was dropped
            m_state = NO_PIECE_PICKED;
            // compute the position of the square and	
            // invoke the grabber to put down
            m_grabber->setPiece(getPositionOfPiece(m_toSquare));
            return;
         } 
      }
   } 
   // The grabber does not have a piece
   // and wants to pick one up
   //-----------------------------------
   else {
      // an nonempty square must be selected
      if (m_squares[selected] == OCCUPIED_FIELD) {

         // a piece will now be picked up
         //-----------------------------------
         // store the current square 
         m_fromSquare = selected; 
         m_toSquare = -1;
         // remember a piece was picked up
         m_state = PIECE_PICKED;

         // compute the position of the piece and invoke
         // the grabber to pick it up
         m_grabber->getPiece(getPositionOfPiece(m_fromSquare));

      } else {
         // a piece cannot be picked up
         m_fromSquare = -1;
         m_toSquare = -1;
      }
      return;
   }

   // in case that an invalid square was selected
   // to drop the piece, we put back the piece to it's
   // original position

   // store the current square
   m_toSquare = m_fromSquare;
   // remember a piece was dropped
   m_state = NO_PIECE_PICKED;
   // compute the position of the square and	
   // invoke the grabber to put down
   m_grabber->setPiece(getPositionOfPiece(m_toSquare));
}

//**********************************************************
//* Gameboard evaluateRules
//**********************************************************
//* This method checks wether the game's rules are honoured.
//* It checks wether the indices of m_toSquare and m_fromSquare
//* are a valid move and returns the index of inbetween piece, 
//* that will be removed, otherwise -1
//**********************************************************
int
Gameboard::evaluateRules()
{
   // compute the coordinates of the squares from the given indices
   int nToY = m_toSquare / 7;
   int nToX = m_toSquare - (7 * nToY);

   int nFromY = m_fromSquare / 7;
   int nFromX = m_fromSquare - (7 * nFromY);

   // compute the distances between the squares
   int distX = nToX - nFromX;
   int distY = nToY - nFromY;
   int nRemoveX, nRemoveY;

   // compute the position of the piece to be removed
   //----------------------------------------------------
   if ((distX == 2) && (distY == 0)) {
      // we're jumping forward in X direction
      nRemoveX = nFromX+1;
      nRemoveY = nFromY;
   } else if  ((distX == -2) && (distY == 0)) {
      // we're jumping backward in X direction
      nRemoveX = nFromX-1;
      nRemoveY = nFromY;
   } else if  ((distX == 0) && (distY == 2)) {
      // we're jumping forward in Y direction
      nRemoveX = nFromX;
      nRemoveY = nFromY+1;
   } else if  ((distX == 0) && (distY == -2)) {
      // we're jumping backward in Y direction
      nRemoveX = nFromX;
      nRemoveY = nFromY-1;
   } else {
      // not a valid move
      return -1;
   }

   // check if the square inbetween is occupied by a piece
   //----------------------------------------------------
   // compute the index of the square
   int removeIndex = nRemoveX + 7 * nRemoveY;
   // check wether it's valid or not
   if (m_squares[removeIndex] == 1) 
      return removeIndex;
   else 
      return -1;
}

//**********************************************************
//* Gameboard getPositionOfPiece
//**********************************************************
//* This method computes the 3D position that the grabber arm
//* should move to, in order to pick up/drop down a piece
//* to/from a square specified by the index
//**********************************************************
SbVec3f
Gameboard::getPositionOfPiece(int index)
{
   // compute the coordinates of the squares from the given indices
   int posY = index / 7;
   int posX = index - (7 * posY);
   // the vector position is computed as in the initSceneGraph method
   // (just with a slightly higher z-axis, since the grabber needs to touch the top of the sphere)
   SbVec3f position(2.*posX, 2.*posY, 2.);
   return position;
}

//**********************************************************
//* Gameboard removePiece
//**********************************************************
//* This method removes the geometry of the piece from the 
//* square specified by index and returns a pointer to the 
//* piece
//**********************************************************
SoNode *
Gameboard::removePiece(int index)
{
   // this should never happen!
   if (m_squares[index] == EMPTY_FIELD || m_squares[index] == INVALID_FIELD)
       return NULL;

   // get the geometry of the piece
   else { //if (m_squares[index] == OCCUPIED_FIELD) {
       SoSeparator *indexField = static_cast<SoSeparator *>(m_sceneGraph->getChild(index));
       // the sphere lies inside the second group of a square
       SoNode *removedSphere = indexField->getChild(1);
       // since we are going to remove the piece, we need to increment the reference counter
       removedSphere->ref();
       // remove geometry
       indexField->removeChild(1);
       // mark square as empty and return sphere
       m_squares[index] = EMPTY_FIELD;
       return removedSphere;
   }

}

//**********************************************************
//* Gameboard insertPiece
//**********************************************************
//* This method inserts geometry of the piece at the 
//* square specified by index into the scenegraph
//**********************************************************
void
Gameboard::insertPiece(int index, SoNode* piece)
{
   // this should never happen!
   if (m_squares[index] == OCCUPIED_FIELD || m_squares[index] == INVALID_FIELD)
       return;

   // insert piece into scenegraph
   else {
       SoSeparator *indexField = static_cast<SoSeparator *>(m_sceneGraph->getChild(index));
       // we hold our pieces just behind their accoirding squares
       indexField->insertChild(piece, 1);
       // mark square as occupied
       m_squares[index] = OCCUPIED_FIELD;
       return;
   }
}
