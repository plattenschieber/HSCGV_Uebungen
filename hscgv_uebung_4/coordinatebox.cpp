/** \file
 * \brief Coordinate system box
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "coordinatebox.h"

#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/Geometry>


CoordinateBox::CoordinateBox(double w, double h, double d)
{
    osg::Geometry *geom = new osg::Geometry;
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 24));
    addDrawable(geom);

    osg::Vec3Array *vert = new osg::Vec3Array;
    vert->push_back(osg::Vec3(0, 0, 0));
    vert->push_back(osg::Vec3(w, 0, 0));
    vert->push_back(osg::Vec3(0, h, 0));
    vert->push_back(osg::Vec3(w, h, 0));
    vert->push_back(osg::Vec3(0, 0, d));
    vert->push_back(osg::Vec3(w, 0, d));
    vert->push_back(osg::Vec3(0, h, d));
    vert->push_back(osg::Vec3(w, h, d));

    vert->push_back(osg::Vec3(0, 0, 0));
    vert->push_back(osg::Vec3(0, h, 0));
    vert->push_back(osg::Vec3(w, 0, 0));
    vert->push_back(osg::Vec3(w, h, 0));
    vert->push_back(osg::Vec3(0, 0, d));
    vert->push_back(osg::Vec3(0, h, d));
    vert->push_back(osg::Vec3(w, 0, d));
    vert->push_back(osg::Vec3(w, h, d));

    vert->push_back(osg::Vec3(0, 0, 0));
    vert->push_back(osg::Vec3(0, 0, d));
    vert->push_back(osg::Vec3(0, h, 0));
    vert->push_back(osg::Vec3(0, h, d));
    vert->push_back(osg::Vec3(w, 0, 0));
    vert->push_back(osg::Vec3(w, 0, d));
    vert->push_back(osg::Vec3(w, h, 0));
    vert->push_back(osg::Vec3(w, h, d));
    geom->setVertexArray(vert);

    osg::Vec4Array *col = new osg::Vec4Array;
    col->push_back(osg::Vec4(1., 0., 0., 1.));
    col->push_back(osg::Vec4(1., 0., 0., 1.));
    for(int i=0; i<6; ++i)
        col->push_back(osg::Vec4(.6, .6, .6, 1.));
    col->push_back(osg::Vec4(0., 1., 0., 1.));
    col->push_back(osg::Vec4(0., 1., 0., 1.));
    for(int i=0; i<6; ++i)
        col->push_back(osg::Vec4(.6, .6, .6, 1.));
    col->push_back(osg::Vec4(0., 0., 1., 1.));
    col->push_back(osg::Vec4(0., 0., 1., 1.));
    for(int i=0; i<6; ++i)
        col->push_back(osg::Vec4(.6, .6, .6, 1.));
    geom->setColorArray(col);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::StateSet *state = getOrCreateStateSet();
    state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::LineWidth *linewidth = new osg::LineWidth(2.);
    state->setAttribute(linewidth, osg::StateAttribute::ON);
    setStateSet(state);
}

CoordinateBox::~CoordinateBox()
{
}
