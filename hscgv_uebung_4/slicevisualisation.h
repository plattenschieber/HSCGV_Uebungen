/** \file
 * \brief Slice visualisation
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef SLICEVISUALISATION_H
#define SLICEVISUALISATION_H

#include <osg/Group>

#include <vector>

#include "slice.h"

class LBMD3Q19;

//! class combining three mutually perpendicular slices through the LBM grid
class SliceVisualisation : public osg::Group
{
    public:
        //! construct from 3D LBM simulation
        SliceVisualisation(LBMD3Q19 *lbm);

        //! set slice to display perpendicular to axis
        void setSlice(int axis, int slice);
        //! get currently displayed slice perpendicular to axis
        int getSlice(int axis) const;

        //! update data/color from simulation using field value
        void update(Slice::DataValue value);
    protected:
        //! dtor
        ~SliceVisualisation();

    private:
        //! reference to LBM simulation, needed for updates
        LBMD3Q19 *m_lbm;

        //! slice numbers for the 3 axes
        std::vector<int> m_sliceNum;
        //! references to the three slices
        std::vector<osg::ref_ptr<Slice> > m_slice;
        //! transform nodes enabling repositioning of slices
        std::vector<osg::ref_ptr<osg::PositionAttitudeTransform> > m_slicePosition;
};

#endif
