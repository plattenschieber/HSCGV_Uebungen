#ifndef GAMEBOARD_H
#define GAMEBOARD_H

/* *************** Programmierpraktikum Computergrafik (CGP) ***************
 * Aufgabe 2 - "Der Greifer"
 *
 * Created by Kai Hormann <Hormann@informatik.uni-erlangen.de>
 * and        Christof Rezk-Salama <Rezk@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: Gameboard.h
 * Description: 
 *   Definition of the "intelligent" gameboard
 */

#include <Inventor/nodes/SoSelection.h>

class Grabber;
class SoNode;
class SbVec3f;
class SoPath;

//! class Gameboard - an intelligent Solitaire gameboard
/*! \anchor Gameboard
 * \b Gameboard is the implementation of an intelligent gameboard, i.e.
 * a gameboard that takes care that the rules are honoured.
 * The gameboard class is meant to be used with the \b Grabber)
 * class to move the pieces around
 * \author   Kai Hormann / Christof Rezk-Salama / Martin Aumueller
 @see \ref Grabber  */

class Gameboard {
   public:
      // ********************************************************************
      // PUBLIC MEMBER FUNCTIONS
      // ********************************************************************

      //! The constructor of class \b Gameboard
      /*! \anchor Constructor
       * The constructor is invoked when creating a new instance of class \b Gameboard
       * using the \b new operator.
       * When creating a gameboard a valid instance of class \b Grabber must be
       * submitted. The \b Gameboard and the \b Grabber communicate using the two
       * functions \b #getPiece,
       * \b #setPiece,
       * \b Grabber::getPiece and
       * \b Grabber::setPiece. */
      Gameboard();

      //! The destructor of class \b Gameboard
      /*! \anchor Destructor The destructor is called when the \b delete
       * operator is invoked to clean up, free the memory allocated by the
       * node and to free all references to scenegraph objects. */
      ~Gameboard();

      //! get the Inventor Scenegraph of the gameboard
      /*!\anchor Gameboard_getSceneGraph
       * The public method getSceneGraph() returns a pointer to an
       * Inventor \b SoSeparator node that contains the scenegraph of the
       * board. Don't forget to call \b ref() and \b unref() if you receive a
       * pointer to a scenegraph object like this!
       */
      SoSeparator *getSceneGraph() {return m_sceneGraph;}

      //! attach a grabber to this gameboard. 
      /*! \anchor Gameboard_attachGrabber
       * This method is called to attach a grabber to this
       * gameboard.
       */
      void attachGrabber(Grabber *grabber);

      //! put down a given piece at the current square of the gamebaord
      /*! \anchor Gameboard_setPiece
       * This function is called by the \b Grabber node, if the \b Grabber has successfully
       * moved the piece to the destination point and wants to put it down. The scenegraph
       * of the piece is given as argument (The \b Grabber node must have previously received
       * this node by calling the #getPiece() method.)
       * By calling this method the \b Grabber tells the \b Gameboard that he has finished to
       * move the piece. From this point on the piece is again part of the \b Gameboard's
       * scenegraph and should be managed only by the \b Gameboard class.
       * The square on the gameboard, onto which the piece is set, is managed entirely by
       * the gameboard's private member variable #m_toSquare. */
      void setPiece(SoNode *piece);

      //! pick up a piece from the current square of the gameboard
      /*! \anchor Gameboard_getPiece
       * This function is called by the Grabber node, if the has moved its arm to the
       * location of the piece and wants to pick it up. Calling this method, the \b Grabber
       * receives a pointer to the scenegraph of the piece, which he can insert into its own
       * scenegraph to move the piece around. From this point on the piece's geometry is part
       * of the grabber's scenegraph and should be managed entirely by the Grabber class.
       * The square on the gamebaord, from which the piece is taken is managed entirely by the 
       * the gameboard's private member variable #m_fromSquare. */
      SoNode *getPiece();

   private:

      // ********************************************************************
      // PRIVATE MEMBER FUNCTIONS
      // ********************************************************************

      //! initalize the gameboard
      /*! \anchor initGameboard
       *  This function is internally called by the \b Gameboard's constructor \ref Constructor
       *  to initalize the internal representation of the gameboard. This function sets the member
       *  array #m_squares to the inital values and implicitly calls the
       *  #initSceneGraph method to initalize the geometry. */
      void initGameboard();


      //! initialize the scenegraph of the gameboard.
      /*! \anchor Gameboard_initSceneGraph
       * This function creates the scenegraph of the entire gameboard and is invoked from 
       * within the function #initGameboard().
       * This method is not supposed to be invoked before a call to the
       * method #initSceneGraph()! */
      void initSceneGraph();

      //! handle the selection of a square
      /*! \anchor selectPiece
       * This funtion handles the selection(picking) of a square. The index of the square
       * is specified by the index argument which refers to the array #m_squares.
       * The function implicitly calls the method #evaluateRules(). */
      void selectPiece(int select);

      //! remove the piece from the square of given index
      /*! \anchor removePiece
       * This method removes and returns the geometry of the piece from the scenegraph.
       */
      SoNode *removePiece(int index);

      //! insert the piece to the square of given index
      /*! \anchor insertPiece
       * This method inserts the geometry piece into the scenegraph.
       */
      void insertPiece(int index, SoNode *piece);

      //! evaluate the rules of the solitaire game
      /*! \anchor evaluateRules
       * This method computes the rules of the Solitaire game.
       * The indices of the squares #m_fromSquare and #m_toSquare, which are involved
       * in the piece's movement are evaluated. If the movement is valid, the function returns
       * the index of the square inbetween, in order to remove the piece from there.
       * If the movement is invalid, the function returns an index of -1.
       */
      int  evaluateRules();

      //! compute the vector position of a piece on a square
      /*! \anchor getPositionOfPiece
       * This method computes the 3D position from/to where the piece should be
       * picked up/dropped down.
       */  
      SbVec3f getPositionOfPiece(int index);

      // ********************************************************************
      // PRIVATE MEMBER VARIABLES
      // ********************************************************************

      //! the possible states of the gameboard
      /*! \anchor GameboardState
       */ 
      enum GameboardState {
         //!  NO_PIECE_PICKED = no piece has been picked up, so we should pick one.
         NO_PIECE_PICKED,
         //!  PIECE_PICKED = a piece was previously picked up and should now be dropped down
         PIECE_PICKED};

      //! the possible states of a single square
      /*! \anchor SquareState
       * These are the values that are contained in the array Gameboard#m_squares
       */
      enum SquareState {
         //! INVALID_FIELD  = square is invalid in this gameboard
         INVALID_FIELD = -1,
         //! EMPTY_FIELD    = square is not occupied by a piece 
         EMPTY_FIELD,
         //! OCCUPIED_FIELD = square is occupied by a piece
         OCCUPIED_FIELD};

      //! an array of integer that represents the squares of the gameboard
      /*! \anchor m_squares
       * This is an array that represents the state of the gameboard's squares squares
       * Posiible values of the integers are:
       * INVALID_FIELD  (= square is invalid in this gameboard),
       * EMPTY_FIELD    (= square is not occupied by a piece) and
       * OCCUPIED_FIELD (= square is occupied by a piece). */
      SquareState m_squares[7*7];


      //! A pointer to the gameboard's scene graph
      /*! This is the root of the gameboard scenegraph. The scene graph is initalized
       * in the method #initSceneGraph. */
      SoSelection *m_sceneGraph;

      //! index of the current square from which a piece is picked up
      /*! \anchor m_fromSquare
       * This integer is used to store the index of the square, from which a
       * piece has been currently picked up.
       @see m_toSquare */
      int m_fromSquare;

      //! index of the current square to which a piece is dropped down
      /*! \anchor m_toSquare
       * This integer is used to store the index of the square, to which a
       * piece has should be dropped down.
       @see m_fromSquare */
      int m_toSquare;

      //! integer that represents the current state of the board
      /*! This integer is used to store the current stat of the gameboard
       * Possible values are:
       *  NO_PIECE_PICKED (= no piece has been picked up, so we should pick one) and
       *  PIECE_PICKED (= a piece was previously picked up and should now be dropped down) */
      GameboardState m_state;

      //! A pointer to the instance of class grabber
      /*! This pointer is used to invoke the grabber to pick up/drop down pieces
       * The pointer is initialized in the constructor \ref Constructor */
      Grabber * m_grabber;
};

#endif
