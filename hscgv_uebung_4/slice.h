/** \file
 * \brief Coloured 2D-slice
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef SLICE_H
#define SLICE_H

#include <osg/Geometry>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/Image>

#include "lbm.h"

//! Coloured 2D-slice
class Slice : public osg::Geometry
{
    public:
        enum DataValue
        {
            Density,
            Velocity,
        };

        //! initialise with an LBM simulation, slice perpendicular to axis
        Slice(LBMD3Q19 *lbm, int axis);
        //! update colours from slice using data field value
        void update(int slice, DataValue value);
    protected:
        //! dtor
        virtual ~Slice();
        //! store reference to simulation for updates
        LBMD3Q19 *m_lbm;
        //! axis perpendicular to slice
        int m_axis;
        //! width of slice in LBM cells
        int m_width;
        //! height of slice in LBM cells
        int m_height;
        //! next largest power of two of m_width
        int m_width2;
        //! next largest power of two of m_height
        int m_height2;
        //! texture holding scalar values
        osg::ref_ptr<osg::Texture2D> m_dataTex;
        //! image holding scalar values
        osg::ref_ptr<osg::Image> m_img;
        //! texture holding 1D transfer function/color map
        osg::ref_ptr<osg::Texture1D> m_tfTex;
};
#endif
