/** \file
 * \brief Line visualisation
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef LINEVISUALISATION_H
#define LINEVISUALISATION_H

#include <osg/Geode>
#include <osg/Array>
#include <osg/Geometry>

#include <vector>

class LBMD3Q19;

//! class showing velocity lines in every LBM cell
class LineVisualisation : public osg::Geode
{
    public:
        //! construct from 3D LBM simulation
        LineVisualisation(LBMD3Q19 *lbm);

        //! update orientation and length from simulation
        void update();
    protected:
        //! dtor
        ~LineVisualisation();

    private:
        //! reference to LBM simulation, needed for updates
        LBMD3Q19 *m_lbm;

        //! width of grid
        int m_width;
        //! height of grid
        int m_height;
        //! depth of grid
        int m_depth;

        //! reference to vertices
        osg::ref_ptr<osg::Vec3Array> m_vert;
        //! reference to geometry object
        osg::ref_ptr<osg::Geometry> m_geom;
};

#endif
