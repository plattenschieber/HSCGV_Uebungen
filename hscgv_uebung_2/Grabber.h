#ifndef GRABBER_H
#define GRABBER_H

/* *************** Programmierpraktikum Computergrafik (CGP) ***************
 * Aufgabe 2 - "Der Greifer"
 *
 * Created by Kai Hormann <Hormann@informatik.uni-erlangen.de>
 * and        Christof Rezk-Salama <Rezk@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: Grabber.h
 * Description: 
 *   Definition of the "grabber" - a robot arm that moves pieces on a gameboard
 */

class Gameboard;
class SbVec3f;
class SoTransform;
class SoRotationXYZ;
class SoMaterial;
class SoSeparator;
class SoGroup;
class SoNode;
class SoSensor;
class SoCalculator;
class SoIdleSensor;

//! class Grabber - a robot arm that moves pieces on a gamebaord
/*! \anchor Grabber
 * \b Grabber is the implementation of an robot arm, that is used together with
 * an instance of class \b Gameboard. The \b Grabber object picks up
 * pieces from definite positions and drops them to other positions
 * \author   Kai Hormann / Christof Rezk-Salama / Martin Aumueller
 @see \ref Gameboard  */

class Grabber {
   public:
      // ********************************************************************
      // PUBLIC MEMBER FUNCTIONS
      // ********************************************************************

      //! The constructor of class \b Grabber
      /*! \anchor Grabber_Constructor
       * The constructor is invoked when creating a new instance of class \b Gameboard
       * using the \b new command. No arguments are required. */
      Grabber();

      //! The destructor of class \b Grabber
      /*! \anchor  Grabber_Destructor
        The destructor is called when the \b delete
       * command is invoked to clean up, free the memory allocated by the
       * node and to free all references to scenegraph objects. */
      ~Grabber();

      //! get the Inventor Scenegraph of the grabber
      /*!\anchor Grabber_getSceneGraph
       * The public method getSceneGraph() returns a pointer to an
       * Inventor \b SoSeparator node that contains the scenegraph of the
       * grabber. Don't forget to call \b ref() and \b unref() if you receive a
       * pointer to a scenegraph object like this!
       */
      SoSeparator *getSceneGraph() {return m_sceneGraph;}

      //! move the arm to the specified position to drop down a piece onto the gamebaord
      /*! \anchor Grabber_setPiece
       * This function is called by the \b Gameboard, to tell the
       * grabber to move his arm to the specified position and drop down a piece.
       * If the arm is in position, the piece is dropped by the grabber by invoking
       * Gameboard::setPiece(..)
       @see Grabber_getPiece Gameboard_setPiece*/
      void setPiece(SbVec3f position);

      //! move the arm to the specified position to pick up a piece from the gamebaord
      /*! \anchor Grabber_getPiece
       * This function is called by the \b Gameboard, to tell the
       * grabber to move his arm to the specified position and pick up a piece.
       * If the arm is in position, the piece is picked up by the grabber by invoking
       * Gameboard::getPiece(..)
       @see Grabber_setPiece Gameboard_getPiece*/
      void getPiece(SbVec3f position);

      //! attach a gameboard to the grabber. 
      /*! \anchor Grabber_attachGameboard
       * This method is called to attach the grabber to a
       * gameboard.
       */
      void attachGameboard(Gameboard *gb);

      //! determine wether the grabber is idle or working.
      bool isWaiting();

   private:
      // ********************************************************************
      // PRIVATE MEMBER FUNCTIONS
      // ********************************************************************

      //! the callback function for the #Grabber's idle sensor
      /*!\anchor Grabber_myIdleCB
       * The static method myIdleCB() handles all movement of the grabber.
       * Here we find out in which state the grabber is and calculate the
       * final state and how much movement is left until then if appropriate.
       * We then perform the requested action.
       */
      static void myIdleCB(void *, SoSensor *);

      //! This function builds the scene graph of the grabber and returns a
      //! pointer to it. It is the caller's responsibility to reference the
      //! scene graph (reference count of the returned node should be 0).
      SoSeparator *initSceneGraph();

      // ********************************************************************
      // PRIVATE MEMBER VARIABLES
      // ********************************************************************

      //! the possible modes of the grabber
      /*! \anchor GrabberMode
       * These are the values that are contained in the variable \b m_mode
       */
      enum GrabberState {
         //! INACTIVE       = grabber is inactive
         INACTIVE,
         //! GET_PIECE      = grabber is performing the animation to pick up a piece
         GET_PIECE,
         //! SET_PIECE      = grabber is performing the animation to drop down the piece
         SET_PIECE
      };

      //! the possible animation phases of the grabber
      /*! \anchor GrabberAnimationPhase
       * These are the values that are contained in the variable \b m_mode
       */
      enum AnimationPhase {
         //! NO_ANIMATION	= no animation in progress
         NO_ANIMATION,
         //! MOVE_GRABBER	= grabber is moving to its position over a piece
         MOVE_GRABBER,
         //! DOWN_GRABBER	= grabber is lowering its under arm
         DOWN_GRABBER,
         //! UP_GRABBER	= grabber is lifting its under arm
         UP_GRABBER
      };

      //! a pointer to our scene graph
      SoSeparator *m_sceneGraph;

      //! a pointer to the gameboard
      Gameboard *m_gameboard;

      //! a pointer to the placeholder group for picked up tiles
      SoGroup *m_pickedUpTile;

      //! the current mode of the grabber
      /*! \anchor Grabber_m_mode
       *  the mode variable determines what the grabber is currently doing
       @see \ref GrabberMode */
      int m_mode;

      //! the current animation phase of the grabber
      /*! \anchor Grabber_m_animationPhase
       *  the mode variable determines what the grabber is currently doing
       @see \ref GrabberMode */
      int m_animationPhase;

      //! a pointer to our idle sensor
      SoIdleSensor *idleSensor;
};

#endif
