/** \file
 * \brief Show coordinate system and grid boundaries
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef COORDBOX_H
#define COORDBOX_H

#include <osg/Geode>

//! show a coordinate system box
class CoordinateBox : public osg::Geode
{
    public:
        //! construct with width w, height h and depth d
        CoordinateBox(double w, double h, double d);
    protected:
        //! dtor
        ~CoordinateBox();
};

#endif
